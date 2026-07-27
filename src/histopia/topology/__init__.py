"""Adaptive 3D topology reconstruction from approved semantic sections."""

from importlib import import_module

_PUBLIC_IMPORTS = {
    "TopologyApproval": ("histopia.topology._approval", "TopologyApproval"),
    "TopologyConfig": ("histopia.topology._config", "TopologyConfig"),
    "approve_topology_result": (
        "histopia.topology._approval",
        "approve_topology_result",
    ),
    "benchmark_topology": (
        "histopia.topology._pipeline",
        "benchmark_topology",
    ),
    "build_topology": ("histopia.topology._pipeline", "build_topology"),
    "load_topology_config": (
        "histopia.topology._config",
        "load_topology_config",
    ),
    "preflight_topology": (
        "histopia.topology._pipeline",
        "preflight_topology",
    ),
    "summarize_topology_run": (
        "histopia.topology._qc",
        "summarize_topology_run",
    ),
    "validate_topology_approval": (
        "histopia.topology._approval",
        "validate_topology_approval",
    ),
    "validate_topology_result": (
        "histopia.topology._result",
        "validate_topology_result",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _PUBLIC_IMPORTS[name]
    except KeyError as error:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = sorted(_PUBLIC_IMPORTS)
