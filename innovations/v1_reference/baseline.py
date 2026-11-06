"""Resolve the unchanged DAGF core in either supported project layout."""
from importlib import import_module
from pathlib import Path


def select_baseline_package(project_root):
    root = Path(project_root)
    candidates = (
        ("dagf_complete", root / "dagf_complete"),
        ("dagf_default_upgrade_20261028.payload.dagf_complete",
         root / "dagf_default_upgrade_20261028" / "payload" / "dagf_complete"),
    )
    for package, directory in candidates:
        if all((directory / (name + ".py")).is_file() for name in ("data", "model", "train")):
            return package
    raise ModuleNotFoundError(
        "Cannot find the completed DAGF core in %s. Supply dagf_complete/ "
        "or dagf_default_upgrade_20261028/payload/dagf_complete/." % root)


BASELINE_PACKAGE = select_baseline_package(Path(__file__).resolve().parents[2])


def load_baseline_module(name):
    return import_module(BASELINE_PACKAGE + "." + name)
