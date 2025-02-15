from histopia import __full_name__, __version__


def test_version_metadata() -> None:
    assert __version__ == "0.1.0.dev0"


def test_full_name_metadata() -> None:
    assert "Histology Spatial Topology" in __full_name__
