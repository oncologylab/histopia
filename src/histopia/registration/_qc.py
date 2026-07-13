"""Clearly labeled registration review images."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from histopia.registration._errors import OptionalDependencyError


def write_labeled_review_panel(
    path: Path | str,
    *,
    panes: list[tuple[str, np.ndarray]],
    title: str,
    metadata: list[str],
) -> Path:
    """Write a labeled horizontal panel without diagnostic checkerboarding."""

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise OptionalDependencyError("pillow", "wsi") from exc

    normalized = [_as_rgb_u8(image) for _, image in panes]
    pane_width = max(image.shape[1] for image in normalized)
    pane_height = max(image.shape[0] for image in normalized)
    header_height = 86
    footer_height = 48 + 18 * len(metadata)
    gap = 8
    canvas = Image.new(
        "RGB",
        (
            pane_width * len(panes) + gap * (len(panes) - 1),
            header_height + pane_height + footer_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((12, 10), title, fill=(20, 24, 28), font=font)
    for index, ((label, _), image) in enumerate(zip(panes, normalized, strict=True)):
        left = index * (pane_width + gap)
        draw.text((left + 8, 48), label, fill=(20, 24, 28), font=font)
        top = header_height + (pane_height - image.shape[0]) // 2
        left_image = left + (pane_width - image.shape[1]) // 2
        canvas.paste(Image.fromarray(image), (left_image, top))
        if index:
            draw.rectangle(
                (left - gap, header_height, left - 1, header_height + pane_height),
                fill=(215, 218, 220),
            )
    footer_top = header_height + pane_height + 16
    for index, line in enumerate(metadata):
        draw.text((12, footer_top + 18 * index), line, fill=(45, 50, 54), font=font)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def write_contact_sheets(
    image_paths: Sequence[Path | str],
    output_dir: Path | str,
    *,
    page_size: int = 6,
    columns: int = 2,
    cell_size: tuple[int, int] = (700, 560),
) -> tuple[Path, ...]:
    """Write labeled review pages without changing source images."""

    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except ImportError as exc:
        raise OptionalDependencyError("pillow", "wsi") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(path) for path in image_paths]
    rows = (page_size + columns - 1) // columns
    label_height = 46
    page_paths: list[Path] = []
    for page_index, start in enumerate(range(0, len(paths), page_size), start=1):
        page = Image.new(
            "RGB",
            (columns * cell_size[0], rows * (cell_size[1] + label_height)),
            "white",
        )
        draw = ImageDraw.Draw(page)
        font = ImageFont.load_default()
        for index, image_path in enumerate(paths[start : start + page_size]):
            row, column = divmod(index, columns)
            left = column * cell_size[0]
            top = row * (cell_size[1] + label_height)
            with Image.open(image_path) as source:
                fitted = ImageOps.contain(source.convert("RGB"), cell_size)
            image_left = left + (cell_size[0] - fitted.width) // 2
            image_top = top + label_height + (cell_size[1] - fitted.height) // 2
            page.paste(fitted, (image_left, image_top))
            draw.text(
                (left + 8, top + 8),
                image_path.stem,
                fill=(20, 24, 28),
                font=font,
            )
        output_path = output_dir / f"mask-audit-{page_index:02d}.jpg"
        page.save(output_path, quality=90)
        page_paths.append(output_path)
    return tuple(page_paths)


def _as_rgb_u8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[:, :, np.newaxis], 3, axis=2)
    array = array[:, :, :3]
    if array.dtype != np.uint8:
        if array.max(initial=0) <= 1.5:
            array = array * 255
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array
