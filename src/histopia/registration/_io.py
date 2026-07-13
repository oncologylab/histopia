"""Image I/O helpers for registration workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from histopia.registration._errors import OptionalDependencyError
from histopia.registration._slides import load_slide_thumbnail


def load_thumbnail(path: Path | str, max_dim_px: int) -> np.ndarray:
    """Load an RGB thumbnail with longest side no larger than ``max_dim_px``."""

    image, _ = load_slide_thumbnail(path, max_dim_px)
    return image


def save_rgb(path: Path | str, image: np.ndarray) -> None:
    """Save an RGB or grayscale array with Pillow."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyError("pillow", "wsi") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a simple red tissue-mask overlay for QC."""

    rgb = _as_uint8_rgb(image).copy()
    mask_bool = np.asarray(mask, dtype=bool)
    rgb[mask_bool, 0] = np.maximum(rgb[mask_bool, 0], 220)
    rgb[mask_bool, 1] = (0.65 * rgb[mask_bool, 1]).astype(np.uint8)
    rgb[mask_bool, 2] = (0.65 * rgb[mask_bool, 2]).astype(np.uint8)
    return rgb


def warp_rgb_thumbnail(
    image: np.ndarray,
    matrix: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Warp an RGB thumbnail into a fixed reference thumbnail frame."""

    try:
        import cv2
    except ImportError as exc:
        raise OptionalDependencyError(
            "opencv-contrib-python-headless",
            "registration",
        ) from exc

    rgb = _as_uint8_rgb(image)
    height, width = output_shape
    return cv2.warpAffine(
        rgb,
        np.asarray(matrix, dtype=np.float32)[:2, :],
        dsize=(width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def warp_mask_thumbnail(
    mask: np.ndarray,
    matrix: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Warp a boolean mask into a fixed reference thumbnail frame."""

    try:
        import cv2
    except ImportError as exc:
        raise OptionalDependencyError(
            "opencv-contrib-python-headless",
            "registration",
        ) from exc
    height, width = output_shape
    warped = cv2.warpAffine(
        (np.asarray(mask, dtype=bool) * 255).astype(np.uint8),
        np.asarray(matrix, dtype=np.float32)[:2, :],
        dsize=(width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped > 0


def blend_rgb(
    reference: np.ndarray,
    moving: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Alpha-blend two RGB thumbnails with matching shape."""

    ref = _as_uint8_rgb(reference).astype(np.float32)
    mov = _as_uint8_rgb(moving).astype(np.float32)
    return np.clip((1 - alpha) * ref + alpha * mov, 0, 255).astype(np.uint8)


def checkerboard_rgb(
    reference: np.ndarray,
    moving: np.ndarray,
    tile_px: int = 48,
) -> np.ndarray:
    """Create a checkerboard QC panel from reference and moving images."""

    ref = _as_uint8_rgb(reference)
    mov = _as_uint8_rgb(moving)
    rows, cols = np.indices(ref.shape[:2])
    selector = ((rows // tile_px) + (cols // tile_px)) % 2 == 0
    out = ref.copy()
    out[selector] = mov[selector]
    return out


def resize_rgb(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize an RGB image by ``scale`` using bilinear interpolation."""

    if scale == 1.0:
        return _as_uint8_rgb(image)
    try:
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyError("pillow", "wsi") from exc

    rgb = _as_uint8_rgb(image)
    height = max(1, int(round(rgb.shape[0] * scale)))
    width = max(1, int(round(rgb.shape[1] * scale)))
    resized = Image.fromarray(rgb).resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(resized)


def resize_mask(mask: np.ndarray, scale: float) -> np.ndarray:
    """Resize a boolean mask by ``scale`` using nearest-neighbor interpolation."""

    if scale == 1.0:
        return np.asarray(mask, dtype=bool)
    try:
        from PIL import Image
    except ImportError as exc:
        raise OptionalDependencyError("pillow", "wsi") from exc

    mask_u8 = (np.asarray(mask, dtype=bool) * 255).astype(np.uint8)
    height = max(1, int(round(mask_u8.shape[0] * scale)))
    width = max(1, int(round(mask_u8.shape[1] * scale)))
    resized = Image.fromarray(mask_u8).resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(resized) > 0


def side_by_side(images: list[np.ndarray], separator_px: int = 8) -> np.ndarray:
    """Concatenate RGB images horizontally with white separators."""

    rgbs = [_as_uint8_rgb(image) for image in images]
    height = max(image.shape[0] for image in rgbs)
    padded = [_pad_to_height(image, height) for image in rgbs]
    separator = np.full((height, separator_px, 3), 255, dtype=np.uint8)
    pieces: list[np.ndarray] = []
    for index, image in enumerate(padded):
        if index:
            pieces.append(separator)
        pieces.append(image)
    return np.concatenate(pieces, axis=1)


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
    arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        if arr.max(initial=0) <= 1.5:
            arr = arr * 255
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    padded = np.full((height, image.shape[1], 3), 255, dtype=np.uint8)
    padded[: image.shape[0], :, :] = image
    return padded
