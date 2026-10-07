"""Observed physical section planes with explicit Z and a 2D fallback."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from histopia._atomic import write_json_atomic, write_text_atomic
from histopia.study._manifest import fingerprint
from histopia.study._serial import validate_serial_study
from histopia.visualization._results_catalog import _copy_asset, _identifier

_ASSETS = Path(__file__).with_name("_serial_stack_assets")


def build_serial_stack_review(
    study,
    planes,
    output_dir,
    *,
    title,
    vendor_dir=None,
    spacing_status="unspecified",
    registration_status="candidate",
):
    """Export observed planes, retaining missing depth and registration limits.

    ``planes`` bind one image per physical section, a common registered frame,
    frame dimensions and XY micrometres per pixel. Repeated channels/restains
    must be composited within their physical plane before calling this function.
    Unknown Z permits section review but disables the physical stack. Display
    exaggeration is explicit and never changes exported Z coordinates.
    """
    if spacing_status not in {"documented", "assumed", "unspecified"}:
        raise ValueError("spacing status must be documented, assumed or unspecified")
    if registration_status not in {"candidate", "previously_reviewed"}:
        raise ValueError("unknown registration status")
    validated = validate_serial_study(study)
    if study.get("fingerprint", validated["fingerprint"]) != validated["fingerprint"]:
        raise ValueError("serial study changed after its fingerprint was frozen")
    sections = {s["physical_section_id"]: s for s in validated["sections"]}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    public, seen, frames, stacks, references = [], set(), set(), set(), set()
    verified = {}
    for plane in planes:
        key = _identifier(plane["physical_section_id"])
        if key in seen or key not in sections:
            raise ValueError("exactly one observed plane per known physical section")
        seen.add(key)
        section = sections[key]
        stacks.add(
            tuple(
                section[k]
                for k in (
                    "source_id",
                    "species",
                    "subject_id",
                    "organ",
                    "specimen_id",
                    "block_id",
                )
            )
        )
        if section.get("z_um") is not None:
            references.add(section["z_reference"])
        mpp = tuple(plane["mpp_xy"])
        shape = tuple(plane["frame_shape_yx"])
        if (
            len(mpp) != 2
            or len(shape) != 2
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v <= 0
                for v in mpp
            )
            or any(
                isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape
            )
        ):
            raise ValueError("frame shape and XY physical scale must be positive")
        frames.add((_identifier(plane["frame_id"]), mpp, shape))
        from PIL import Image

        with Image.open(plane["image_path"]) as image:
            if image.size != tuple(reversed(shape)):
                raise ValueError("image dimensions differ from the registered frame")
        image = _copy_asset(
            {"path": plane["image_path"], "sha256": plane["image_sha256"]},
            output,
            verified,
        )
        public.append(
            {
                "id": key,
                "order": section["section_order"],
                "z_um": section.get("z_um"),
                "z_evidence": section.get("z_evidence"),
                "modality": str(plane["modality"]),
                "image": image,
                "mpp_xy": list(mpp),
                "frame_shape_yx": list(shape),
                "registration": plane.get("registration", {}),
                "upstream_fingerprint": plane["upstream_fingerprint"],
                "observed": True,
            }
        )
    if not public or len(stacks) != 1 or len(frames) != 1 or len(references) > 1:
        raise ValueError("planes must share one specimen/block, frame and Z reference")
    public.sort(key=lambda p: p["order"])
    first = sections[public[0]["id"]]
    payload = {
        "schema_version": "serial-review-1",
        "title": str(title),
        "study_fingerprint": validated["fingerprint"],
        "source_id": first["source_id"],
        "organ": first["organ"],
        "subject_id": first["subject_id"],
        "specimen_id": first["specimen_id"],
        "planes": public,
        "physical_stack_available": all(p["z_um"] is not None for p in public),
        "spacing_status": spacing_status,
        "registration_status": registration_status,
        "coordinate_units": "µm",
        "interpolated_planes": 0,
        "scientific_approval": False,
    }
    payload["fingerprint"] = fingerprint(payload)
    write_json_atomic(output / "stack.json", payload)
    write_text_atomic(
        output / "stack-data.js",
        "globalThis.HISTOPIA_SERIAL_STACK="
        + json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
        + ";\n",
    )
    for name in ("index.html", "stack.css", "stack.js"):
        write_text_atomic(output / name, (_ASSETS / name).read_text())
    if vendor_dir is not None:
        (output / "vendor").mkdir(exist_ok=True)
        for name in ("three.module.min.js", "OrbitControls.js"):
            shutil.copyfile(Path(vendor_dir) / name, output / "vendor" / name)
    return output / "index.html"
