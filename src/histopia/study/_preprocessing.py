"""Explicit modality-correct RGB input for new H&E feature qualification."""

from __future__ import annotations

import numpy as np

HE_RGB_POLICY = {
    "id": "he-native-rgb-v1",
    "modality": "H&E",
    "stain_deconvolution": "none",
    "color_normalization": "none",
    "encoder_normalization": "checkpoint-transform",
    "qualification": "bounded-fields-required-before-whole-slide",
}


def prepare_he_rgb(image: np.ndarray) -> np.ndarray:
    """Retain H&E color, followed by the encoder's own checkpoint transform.

    This fixed, fit-free policy avoids estimating a DAB concentration from
    eosin. It is a separate feature identity from the legacy H–DAB projection;
    it does not imply that legacy models are compatible with the new domain.
    """
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("H&E preprocessing requires uint8 RGB input")
    return rgb.copy()
