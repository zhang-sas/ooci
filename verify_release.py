"""Check release hashes, local imports and Python syntax without model execution."""
import ast
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "release_manifest.json").read_text(encoding="utf-8"))
    failures = []
    actual = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file()
              and ".git" not in path.relative_to(ROOT).parts and "__pycache__" not in path.parts
              and path.name != "release_manifest.json" and path.suffix != ".pyc"
              and not any(part.startswith("runs") for part in path.relative_to(ROOT).parts)
              and not path.relative_to(ROOT).parts[0] in ("roberta", "dataset", "opinion-lexicon-English")}
    expected = set(manifest["files"])
    # External-asset directories contain tracked README files only.
    for relative in expected:
        path = ROOT / relative
        if not path.is_file():
            failures.append("Missing release file: " + relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][relative]:
            failures.append("Modified release file: " + relative)
    for relative in sorted(actual - expected):
        failures.append("Unlisted code/document: " + relative)
    modules = {}
    for path in ROOT.rglob("*.py"):
        if any(part in (".git", "__pycache__") or part.startswith("runs") for part in path.relative_to(ROOT).parts):
            continue
        parts = list(path.relative_to(ROOT).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        modules[".".join(parts)] = path
    for name, path in modules.items():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path), feature_version=(3, 8))
        except SyntaxError as error:
            failures.append(str(error))
            continue
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    base = ".".join(parts[:len(parts) - node.level + 1] + ([base] if base else []))
                if base.split(".")[0] in ("aeor_rg", "innovations", "dagf_complete") and base not in modules:
                    failures.append("Unresolved local import: " + name + " -> " + base)
            elif isinstance(node, ast.Import):
                for item in node.names:
                    if item.name.split(".")[0] in ("aeor_rg", "innovations", "dagf_complete") and item.name not in modules:
                        failures.append("Unresolved local import: " + name + " -> " + item.name)
    from aeor_rg.runtime import config_for
    for dataset in ("restaurant", "laptop", "mams", "twitter"):
        config_for(dataset)
    provenance = json.loads((ROOT / "source_provenance.json").read_text(encoding="utf-8"))
    for relative, record in provenance["copied_files"].items():
        if not (ROOT / relative).is_file() or hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != record["sha256"]:
            failures.append("Frozen source changed: " + relative)
    result = dict(passed=not failures, files_checked=len(expected), python_modules_checked=len(modules),
                  failures=failures, training_performed=False, model_inference_performed=False,
                  tensor_runtime_checked=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
