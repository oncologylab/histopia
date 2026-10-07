"""Bounded Human Protein Atlas metadata adapter; no implied TCGA matching."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from histopia.study._manifest import file_sha256


def hpa_ihc_inventory(
    xml_path: Path | str, *, organs=("pancreas", "liver", "lung", "kidney")
) -> list[dict[str, object]]:
    """Read antibody-specific normal human tissue IHC records from HPA XML.

    Retain antibody, antigen, tissue, donor and image identity. ``specimen_id``
    is a legacy alias for HPA's patientId, not a verified block or tissue core.
    Repeated patient IDs do not establish serial order, section spacing or
    same-cell identity. These are
    external human IHC exemplars, not matched H&E/TCGA cell outcomes. Antibody
    epitopes and phospho-vs-total equivalence need a separate curated mapping.
    """
    root = ET.parse(xml_path).getroot()
    digest = file_sha256(xml_path)
    rows = []
    for entry in root.findall("entry"):
        gene = entry.findtext("name")
        identifier = entry.find("identifier")
        if identifier is None:
            continue
        for antibody in entry.findall("antibody"):
            for expression in antibody.findall("tissueExpression"):
                if (
                    expression.get("technology") != "IHC"
                    or expression.get("assayType") != "tissue"
                ):
                    continue
                for data in expression.findall("data"):
                    tissue = data.findtext("tissue", "").lower()
                    if tissue not in organs:
                        continue
                    annotation = [
                        {
                            "cell_type": cell.findtext("cellType"),
                            "staining": cell.findtext("level[@type='staining']"),
                            "intensity": cell.findtext("level[@type='intensity']"),
                            "quantity": cell.findtext("quantity"),
                        }
                        for cell in data.findall("tissueCell")
                    ]
                    for patient in data.findall("patient"):
                        for image in patient.findall(".//imageUrl"):
                            rows.append(
                                {
                                    "gene": gene,
                                    "ensembl_id": identifier.get("id"),
                                    "species": "Homo sapiens",
                                    "organ": tissue,
                                    "antibody_id": antibody.get("id"),
                                    "specimen_id": patient.findtext("patientId"),
                                    "donor_id": patient.findtext("patientId"),
                                    "specimen_id_kind": "HPA patientId (donor)",
                                    "image_url_tif": next(
                                        (
                                            candidate.text
                                            for parent in patient.findall(".//image")
                                            if parent.find("imageUrl") is image
                                            for candidate in parent.findall(
                                                "imageUrlTif"
                                            )
                                        ),
                                        None,
                                    ),
                                    "antibody_validation": expression.findtext(
                                        "verification[@type='validation']"
                                    ),
                                    "core_id": None,
                                    "block_id": None,
                                    "section_order": None,
                                    "z_spacing_um": None,
                                    "matched_he_image_url": None,
                                    "pairing_status": "unverified",
                                    "image_url": image.text,
                                    "annotation": annotation,
                                    "source_xml_sha256": digest,
                                    "phospho_total_equivalence": "unverified",
                                    "matched_tcga_target": False,
                                    "role": "external_ihc_reference_pilot",
                                }
                            )
    return rows
