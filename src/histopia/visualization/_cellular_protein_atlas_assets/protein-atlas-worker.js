const cache = new Map();
const CACHE_LIMIT = 64;

async function values(url, count) {
  if (!url) return null;
  if (cache.has(url)) {
    const cached = cache.get(url);
    cache.delete(url);
    cache.set(url, cached);
    return cached;
  }
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok) throw new Error("Could not load a protein value chunk");
  const buffer = await response.arrayBuffer();
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 5));
  if (
    magic !== "HCPA1" ||
    ![3, 4].includes(view.getUint8(5)) ||
    view.getUint8(6) !== 1
  ) {
    throw new Error("Invalid HCPA1 protein values");
  }
  const cells = view.getUint32(8, true);
  const header = view.getUint32(12, true);
  if (cells !== count) throw new Error("Protein values and cell geometry differ");
  const row = new Uint8Array(buffer, header, cells);
  cache.set(url, row);
  while (cache.size > CACHE_LIMIT) cache.delete(cache.keys().next().value);
  return row;
}

function rgb(hex) {
  return [
    parseInt(hex.slice(1, 3), 16) / 255,
    parseInt(hex.slice(3, 5), 16) / 255,
    parseInt(hex.slice(5, 7), 16) / 255,
  ];
}

function decoded(value) {
  return value ? (value - 1) / 254 : null;
}

function displayIntensity(channel, index, data) {
  let intensity;
  if (data.expression === "residual") {
    const predicted = decoded(channel.predicted[index]);
    const observed = channel.observed ? decoded(channel.observed[index]) : null;
    if (predicted === null || observed === null) return 0;
    intensity = Math.abs(predicted - observed);
  } else {
    const source =
      data.expression === "observed" ? channel.observed : channel.predicted;
    const raw = source ? decoded(source[index]) : null;
    if (raw === null) return 0;
    if (data.patternNormalized) {
      const lower = Number(channel.transform.lower_fraction) || 0;
      const upper = Number(channel.transform.upper_fraction) || 1;
      const gamma = Number(channel.transform.gamma) || 1;
      intensity = Math.pow(
        Math.max(0, Math.min(1, (raw - lower) / Math.max(upper - lower, 1e-6))),
        gamma,
      );
    } else {
      intensity = raw;
    }
  }
  return Math.max(0, intensity - data.threshold) / Math.max(1 - data.threshold, 1e-3);
}

self.onmessage = async (event) => {
  const data = event.data;
  try {
    const rows = [];
    for (const section of data.sections) {
      const channels = [];
      for (const channel of section.channels) {
        const predicted = await values(channel.predicted, section.count);
        const observed = channel.observed
          ? await values(channel.observed, section.count)
          : null;
        channels.push({
          ...channel,
          rgb: rgb(channel.color),
          predicted,
          observed,
        });
      }
      const colors = new Uint8Array(section.count * 4);
      for (let index = 0; index < section.count; index += 1) {
        let best = 0;
        let second = 0;
        let bestColor = null;
        let secondColor = null;
        for (const channel of channels) {
          const intensity = displayIntensity(channel, index, data);
          if (intensity > best) {
            second = best;
            secondColor = bestColor;
            best = intensity;
            bestColor = channel.rgb;
          } else if (intensity > second) {
            second = intensity;
            secondColor = channel.rgb;
          }
        }
        if (!bestColor || best <= 0) continue;
        const brightness = 0.3 + 0.7 * Math.min(best, 1);
        let red = bestColor[0] * brightness;
        let green = bestColor[1] * brightness;
        let blue = bestColor[2] * brightness;
        if (data.composite === "overlap" && secondColor && second > 0) {
          const contribution = Math.min(second, best) * 0.28;
          red += secondColor[0] * contribution;
          green += secondColor[1] * contribution;
          blue += secondColor[2] * contribution;
          const maximum = Math.max(red, green, blue, 1);
          red /= maximum;
          green /= maximum;
          blue /= maximum;
        }
        colors[4 * index] = Math.round(Math.min(red, 1) * 255);
        colors[4 * index + 1] = Math.round(Math.min(green, 1) * 255);
        colors[4 * index + 2] = Math.round(Math.min(blue, 1) * 255);
        colors[4 * index + 3] = Math.round(Math.min(0.22 + best * 0.88, 1) * 255);
      }
      rows.push({ id: section.id, colors: colors.buffer });
    }
    self.postMessage(
      { request: data.request, rows },
      rows.map((row) => row.colors),
    );
  } catch (error) {
    self.postMessage({ request: data.request, error: error.message || String(error) });
  }
};
