"use strict";

const reviewData = globalThis.HISTOPIA_REVIEW_MANIFEST;
const feedbackConfig = reviewData?.feedback;
const feedbackPanel = document.querySelector("#registration-feedback");
if (!reviewData || !feedbackConfig || !feedbackPanel) {
  throw new Error("Missing Histopia registration feedback configuration");
}

const labelNames = {
  missing_tissue: "Missing tissue",
  extra_debris: "Extra debris",
  glass_border: "Glass border",
  stain_artifact: "Stain artifact",
  internal_holes: "Internal holes",
  excess_whitespace: "Excess whitespace",
  fragmented_tissue: "Fragmented tissue",
  wrong_orientation: "Wrong orientation",
  wrong_position: "Wrong position",
  abrupt_morphology_jump: "Abrupt morphology jump",
  anchor_issue: "Anchor issue",
  duplicate_section: "Duplicate section",
  missing_section: "Missing section",
  global_shift: "Global shift",
  rotation_error: "Rotation error",
  scale_error: "Scale error",
  local_misalignment: "Local misalignment",
  poor_overlap: "Poor overlap",
  crop_error: "Crop error",
  wrong_reference: "Wrong reference",
  other: "Other",
};
feedbackPanel.innerHTML = `
  <h2 id="feedback-title">Slide review</h2>
  <span id="feedback-status" class="feedback-status">Loading feedback</span>
  <div id="feedback-access" class="access">
    <input id="feedback-key" type="password" placeholder="Access key"
      autocomplete="current-password">
    <button id="feedback-connect" type="button">Connect</button>
  </div>
  <div class="navigation">
    <button id="feedback-previous" type="button">Previous</button>
    <button id="feedback-next" type="button">Next open</button>
  </div>
  <div class="decisions">
    <button type="button" data-feedback-decision="accept">Accept</button>
    <button type="button" data-feedback-decision="hold">Hold</button>
    <button type="button" data-feedback-decision="reject">Reject</button>
  </div>
  <fieldset id="feedback-labels"><legend>Issues</legend></fieldset>
  <div id="feedback-order" class="order-corrections" hidden>
    <label>Suggested order<input id="feedback-suggested-order" type="number" min="1"></label>
    <label>Rotation<select id="feedback-rotation">
      <option value="">Unchanged</option><option value="0">0 degrees</option>
      <option value="1">90 degrees CCW</option><option value="2">180 degrees</option>
      <option value="3">270 degrees CCW</option>
    </select></label>
  </div>
  <label>Reviewer<input id="feedback-reviewer" autocomplete="name" required></label>
  <label>Comment<textarea id="feedback-comment" maxlength="4000"></textarea></label>
  <button id="feedback-save" class="save" type="button" disabled>Save slide review</button>
  <span id="feedback-message" class="message" role="status"></span>`;

const feedbackElements = {
  title: document.querySelector("#feedback-title"),
  status: document.querySelector("#feedback-status"),
  access: document.querySelector("#feedback-access"),
  key: document.querySelector("#feedback-key"),
  connect: document.querySelector("#feedback-connect"),
  labels: document.querySelector("#feedback-labels"),
  order: document.querySelector("#feedback-order"),
  suggestedOrder: document.querySelector("#feedback-suggested-order"),
  rotation: document.querySelector("#feedback-rotation"),
  reviewer: document.querySelector("#feedback-reviewer"),
  comment: document.querySelector("#feedback-comment"),
  save: document.querySelector("#feedback-save"),
  message: document.querySelector("#feedback-message"),
};
const decisionButtons = [...document.querySelectorAll("[data-feedback-decision]")];
const cards = [...document.querySelectorAll("#slides article")];
cards.forEach((card, index) => {
  card.dataset.feedbackSlide = reviewData.slides[index].slide;
  card.tabIndex = 0;
});
let feedback = null;
let currentSlideId = reviewData.slides[0].slide;
let currentDecision = "";
let authenticationRequired = true;
feedbackElements.key.value = sessionStorage.getItem("histopiaReviewKey") || "";
feedbackElements.reviewer.value = sessionStorage.getItem("histopiaReviewer") || "";
feedbackElements.order.hidden = feedbackConfig.stage !== "order";
feedbackElements.suggestedOrder.max = String(reviewData.slides.length);

function authorizationHeaders() {
  const headers = {"Content-Type": "application/json"};
  if (authenticationRequired) {
    headers.Authorization = `Bearer ${feedbackElements.key.value}`;
  }
  return headers;
}

async function configureFeedbackAccess() {
  if (location.protocol === "file:") {
    feedbackElements.status.textContent = "Static review";
    return;
  }
  const response = await fetch("/api/reviews/access", {cache: "no-store"});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Review service unavailable");
  if (!payload.review_configured) {
    feedbackElements.status.textContent = "Feedback unavailable";
    return;
  }
  authenticationRequired = Boolean(payload.authentication_required);
  feedbackElements.access.hidden = !authenticationRequired;
  if (!authenticationRequired || feedbackElements.key.value) {
    await connectFeedback();
  } else {
    feedbackElements.status.textContent = "Connect to load feedback";
  }
}

function currentRecord() {
  return feedback?.feedback?.[currentSlideId] || null;
}

function selectSlide(slideId) {
  currentSlideId = slideId;
  const row = reviewData.slides.find((slide) => slide.slide === slideId);
  const record = currentRecord();
  feedbackElements.title.textContent =
    `${String(row.order).padStart(2, "0")} ${row.label}`;
  currentDecision = record?.decision || "";
  decisionButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.feedbackDecision === currentDecision);
  });
  feedbackElements.labels.querySelectorAll("input").forEach((input) => {
    input.checked = Boolean(record?.labels?.includes(input.value));
  });
  feedbackElements.comment.value = record?.comment || "";
  feedbackElements.suggestedOrder.value = record?.suggested_order || "";
  feedbackElements.rotation.value =
    record?.suggested_quarter_turns_ccw == null
      ? "" : String(record.suggested_quarter_turns_ccw);
  cards.forEach((card) => {
    card.classList.toggle("feedback-selected", card.dataset.feedbackSlide === slideId);
  });
}

function renderFeedback() {
  const latest = feedback.feedback || {};
  cards.forEach((card) => {
    const record = latest[card.dataset.feedbackSlide];
    card.classList.remove("feedback-accept", "feedback-hold", "feedback-reject");
    if (record) card.classList.add(`feedback-${record.decision}`);
  });
  feedbackElements.status.textContent =
    `${feedback.summary.reviewed}/${feedback.slides.length} reviewed`;
  feedbackElements.save.disabled = false;
  selectSlide(currentSlideId);
}

async function connectFeedback() {
  feedbackElements.message.textContent = "";
  const query = new URLSearchParams({
    cohort: feedbackConfig.cohort,
    stage: feedbackConfig.stage,
  });
  const response = await fetch(`/api/reviews/feedback?${query}`, {
    headers: authorizationHeaders(),
    cache: "no-store",
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Unable to load feedback");
  if (payload.fingerprint !== reviewData.fingerprint) {
    throw new Error("Displayed review is stale; rebuild it before recording feedback");
  }
  feedback = payload;
  if (authenticationRequired) {
    sessionStorage.setItem("histopiaReviewKey", feedbackElements.key.value);
  }
  feedbackElements.labels.querySelectorAll("label").forEach((label) => label.remove());
  for (const labelId of feedback.labels) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = labelId;
    label.append(input, document.createTextNode(labelNames[labelId] || labelId));
    feedbackElements.labels.append(label);
  }
  renderFeedback();
}

function moveSlide(direction, openOnly = false) {
  let rows = reviewData.slides;
  if (openOnly && feedback) {
    rows = rows.filter((slide) => !feedback.feedback?.[slide.slide]);
  }
  if (!rows.length) rows = reviewData.slides;
  const index = rows.findIndex((slide) => slide.slide === currentSlideId);
  const next = index < 0 ? 0 : (index + direction + rows.length) % rows.length;
  selectSlide(rows[next].slide);
}

decisionButtons.forEach((button) => button.addEventListener("click", () => {
  currentDecision = button.dataset.feedbackDecision;
  if (currentDecision === "accept") {
    feedbackElements.labels.querySelectorAll("input").forEach((input) => {
      input.checked = false;
    });
  }
  decisionButtons.forEach((row) => {
    row.classList.toggle("active", row === button);
  });
  feedbackElements.message.style.color = "#52605a";
  feedbackElements.message.textContent =
    `${button.textContent} selected. Enter a reviewer and save this slide review.`;
}));
cards.forEach((card) => {
  card.addEventListener("click", () => selectSlide(card.dataset.feedbackSlide));
  card.addEventListener("keydown", (event) => {
    if (event.key === "Enter") selectSlide(card.dataset.feedbackSlide);
  });
});
feedbackElements.connect.addEventListener("click", () => {
  connectFeedback().catch((error) => {
    feedbackElements.message.textContent = error.message;
  });
});
feedbackElements.key.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    feedbackElements.connect.click();
  }
});
document.querySelector("#feedback-previous").addEventListener("click", () => {
  moveSlide(-1);
});
document.querySelector("#feedback-next").addEventListener("click", () => {
  moveSlide(1, true);
});
feedbackElements.save.addEventListener("click", async () => {
  feedbackElements.message.textContent = "";
  if (!currentDecision) {
    feedbackElements.message.textContent = "Choose Accept, Hold, or Reject first.";
    decisionButtons[0].focus();
    return;
  }
  const reviewer = feedbackElements.reviewer.value.trim();
  if (!reviewer) {
    feedbackElements.message.textContent =
      "Enter a reviewer name before saving.";
    feedbackElements.reviewer.focus();
    return;
  }
  sessionStorage.setItem("histopiaReviewer", feedbackElements.reviewer.value);
  const payload = {
    cohort: feedbackConfig.cohort,
    stage: feedbackConfig.stage,
    fingerprint: feedback.fingerprint,
    slide_id: currentSlideId,
    decision: currentDecision,
    labels: [...feedbackElements.labels.querySelectorAll("input:checked")]
      .map((input) => input.value),
    reviewer,
    comment: feedbackElements.comment.value,
  };
  if (feedbackConfig.stage === "order") {
    if (feedbackElements.suggestedOrder.value) {
      payload.suggested_order = Number(feedbackElements.suggestedOrder.value);
    }
    if (feedbackElements.rotation.value !== "") {
      payload.suggested_quarter_turns_ccw = Number(feedbackElements.rotation.value);
    }
  }
  feedbackElements.save.disabled = true;
  try {
    const response = await fetch("/api/reviews/feedback", {
      method: "POST",
      headers: authorizationHeaders(),
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Unable to save feedback");
    feedback = result.feedback;
    feedbackElements.message.style.color = "#146b46";
    feedbackElements.message.textContent = "Slide review saved";
    renderFeedback();
  } catch (error) {
    feedbackElements.message.style.color = "#8e2623";
    feedbackElements.message.textContent = error.message;
    feedbackElements.save.disabled = false;
  }
});
selectSlide(currentSlideId);
configureFeedbackAccess().catch((error) => {
  feedbackElements.status.textContent = "Feedback unavailable";
  feedbackElements.message.textContent = error.message;
});
