from pathlib import Path

from histopia.registration import build_kpf_manifest, normalize_slide_stem


def test_normalize_slide_stem_uses_bracket_id() -> None:
    assert normalize_slide_stem("[#042] Yi_#4577_panc_cJun.ndpi") == "slide-0042"
    assert normalize_slide_stem("[#042] Yi_#4577_panc_cJun.ome.tiff") == "slide-0042"
    assert (
        normalize_slide_stem("Yi_#4630-panc_HE-[350]-collection_0000046596_2017.scn")
        == "marker-he"
    )
    assert (
        normalize_slide_stem("Yi_#4257Panc_Yap(rab)-[9]-collection_0000046151_2017.scn")
        == "marker-yap-rab"
    )


def test_build_kpf_manifest_pairs_raw_and_reference(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_wsi"
    registered_dir = tmp_path / "registered"
    raw_dir.mkdir()
    registered_dir.mkdir()
    (raw_dir / "[#042] Yi_#4577_panc_cJun.ndpi").touch()
    (registered_dir / "[#042] Yi_#4577_panc_cJun.ome.tiff").touch()
    (raw_dir / "[#043] Yi_#4577_panc_HE.scn").touch()

    manifest = build_kpf_manifest(tmp_path)

    assert len(manifest.pairs) == 1
    assert manifest.pairs[0].key == "slide-0042"
    assert manifest.missing_reference_keys == ("marker-he",)
    assert manifest.missing_raw_keys == ()
