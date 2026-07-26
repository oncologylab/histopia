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


def test_build_kpf_manifest_disambiguates_reused_scanner_ids(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw_wsi"
    registered_dir = tmp_path / "registered"
    raw_dir.mkdir()
    registered_dir.mkdir()
    for marker in ("JunB", "Nr2f1"):
        (raw_dir / f"[#019] Yi_#4314_panc_{marker}.ndpi").touch()
        (registered_dir / f"23_[#019] Yi_#4314_panc_{marker}.ome.tiff").touch()

    manifest = build_kpf_manifest(tmp_path)

    assert manifest.is_complete
    assert [pair.key for pair in manifest.pairs] == [
        "slide-0019-marker-junb",
        "slide-0019-marker-nr2f1",
    ]
    assert [pair.raw_path.stem.rsplit("_", 1)[-1] for pair in manifest.pairs] == [
        "JunB",
        "Nr2f1",
    ]


def test_build_kpf_manifest_retains_genuinely_ambiguous_marker(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw_wsi"
    registered_dir = tmp_path / "registered"
    raw_dir.mkdir()
    registered_dir.mkdir()
    (raw_dir / "[#019] Yi_#4314_panc_JunB.ndpi").touch()
    (raw_dir / "copy_[#019] Yi_#4314_panc_JunB.ndpi").touch()
    (registered_dir / "[#019] Yi_#4314_panc_JunB.ome.tiff").touch()

    manifest = build_kpf_manifest(tmp_path)

    assert not manifest.is_complete
    assert manifest.pairs == ()
    assert manifest.ambiguous_keys == ("slide-0019",)


def test_build_kpf_manifest_ignores_registered_mask_artifacts(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw_wsi"
    registered_dir = tmp_path / "registered"
    raw_dir.mkdir()
    registered_dir.mkdir()
    (raw_dir / "[#076] Yi_#6134_panc_CK19.ndpi").touch()
    (registered_dir / "[#076] Yi_#6134_panc_CK19.ome.tiff").touch()
    (registered_dir / "[#076] Yi_#6134_panc_CK19.ome_mask.tif").touch()
    (registered_dir / "[#076] Yi_#6134_panc_CK19_mask.tiff").touch()

    manifest = build_kpf_manifest(tmp_path)

    assert manifest.is_complete
    assert len(manifest.pairs) == 1
    assert manifest.pairs[0].reference_path.name.endswith(".ome.tiff")
