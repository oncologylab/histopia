from __future__ import annotations

import numpy as np
import pytest

from histopia.cells._tile_profiles import profile_tile_instances
from histopia.study._external import hpa_ihc_inventory


def test_sparse_instance_ids_support_pixel_centers_and_anisotropic_units():
    raw = np.array([[0, 99, 99], [7, 0, 0], [7, 0, 3_000_000]], dtype=np.uint32)
    supported = raw.copy()
    supported[0, 2] = supported[2, 2] = 0
    p = profile_tile_instances(raw, supported, origin_xy=(100, 200), mpp_xy=(0.5, 2))
    np.testing.assert_array_equal(p["label_id"], [7, 99, 3_000_000])
    np.testing.assert_array_equal(p["raw_area_px"], [2, 2, 1])
    np.testing.assert_allclose(p["supported_fraction"], [1, 0.5, 0])
    np.testing.assert_allclose(p["native_xy_px"][:2], [[100.5, 202], [101.5, 200.5]])
    np.testing.assert_allclose(p["native_xy_um"][:2], [[50.25, 404], [50.75, 401]])
    np.testing.assert_allclose(p["supported_area_um2"][:2], [2, 1])
    assert np.isnan(p["native_xy_px"][2]).all()
    assert np.isnan(p["supported_area_um2"][2])
    assert p["touches_tile_edge"].all()


def test_unknown_scale_empty_instances_and_support_validation():
    raw = np.zeros((3, 3), np.uint32)
    assert profile_tile_instances(raw, raw)["native_xy_px"].shape == (0, 2)
    raw[1, 1] = 4
    p = profile_tile_instances(raw, raw)
    assert not p["touches_tile_edge"].any()
    assert np.isnan(p["native_xy_um"]).all()
    assert np.isnan(p["supported_area_um2"]).all()
    wrong = raw.copy()
    wrong[1, 1] = 8
    with pytest.raises(ValueError, match="preserve raw label IDs"):
        profile_tile_instances(raw, wrong)
    for scale in [(0, 1), (-1, 1), (np.nan, 1), (1,)]:
        with pytest.raises(ValueError, match="pixel spacings"):
            profile_tile_instances(raw, raw, mpp_xy=scale)


def test_hpa_donor_match_does_not_create_serial_or_same_section_pairing(tmp_path):
    path = tmp_path / "entry.xml"
    path.write_text("""<proteinAtlas><entry><name>KRT19</name><identifier id="gene"/>
      <antibody id="ab"><tissueExpression technology="IHC" assayType="tissue">
      <verification type="validation">enhanced</verification>
      <data><tissue>Liver</tissue>
      <patient><patientId>123</patientId><sample><assayImage><image>
      <imageUrl>https://images.proteinatlas.org/a.jpg</imageUrl>
      <imageUrlTif>https://images.proteinatlas.org/a.tif</imageUrlTif>
      </image></assayImage></sample></patient></data></tissueExpression></antibody>
      </entry></proteinAtlas>""")
    row = hpa_ihc_inventory(path)[0]
    assert row["donor_id"] == row["specimen_id"] == "123"
    assert row["image_url_tif"].endswith("a.tif")
    assert row["antibody_validation"] == "enhanced"
    assert row["pairing_status"] == "unverified"
    for key in (
        "core_id",
        "block_id",
        "section_order",
        "z_spacing_um",
        "matched_he_image_url",
    ):
        assert row[key] is None
