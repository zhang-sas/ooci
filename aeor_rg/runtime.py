"""Configuration and artifact utilities; safe to import without PyTorch."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("restaurant", "laptop", "mams", "twitter")
PAPER_DATASETS = DATASETS[:3]
PATH_KEYS = ("train_file", "test_file", "roberta_path", "lexicon_dir")
FULL_FLAGS = dict(comparison_profile="full", relation_reweight=True, readout="cls",
                  baseline_regularization_weight=0., no_routing=False, no_shared=False,
                  no_inter_aspect=False, no_structure_prior=False, no_amr=False,
                  no_agf=False, no_lexicon=False, evaluation_protocol="baseline", max_steps=0)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def canonical_dataset(name):
    return {"res": "restaurant", "lap": "laptop"}.get(name, name)


def require_full(config):
    for key, wanted in FULL_FLAGS.items():
        if config.get(key) != wanted:
            raise ValueError("Full-model configuration mismatch: " + key)
    if config.get("val_file") is not None:
        raise ValueError("This release preserves full-train / periodic-test selection")


def resolve_backbone(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not (path / "config.json").is_file() and (path / "roberta/config.json").is_file():
        path = path / "roberta"
    return path.resolve()


def config_for(dataset, seed=1000, overrides=None):
    dataset = canonical_dataset(dataset)
    if dataset not in DATASETS or seed < 0:
        raise ValueError("Unknown dataset or negative seed")
    config = read_json(ROOT / "configs" / (dataset + ".json"))
    config["seed"] = seed
    for key, value in (overrides or {}).items():
        if value is not None:
            if key not in config and key not in ("threads", "num_workers"):
                raise ValueError("Unsupported configuration override: " + key)
            config[key] = value
    for key in PATH_KEYS:
        path = Path(config[key]).expanduser()
        config[key] = str((path if path.is_absolute() else ROOT / path).resolve())
    config["roberta_path"] = str(resolve_backbone(config["roberta_path"]))
    require_full(config)
    integer_positive = ("epochs", "batch_size", "eval_batch_size", "log_step", "grad_accum_steps",
                        "max_length", "max_amr_edges", "latency_repeats")
    for key in integer_positive:
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if min(config["lr"], config["added_lr"]) <= 0 or config["weight_decay"] < 0:
        raise ValueError("Invalid learning rate or weight decay")
    if config.get("num_workers", 0) < 0 or config.get("threads", 0) < 0 or config["latency_sentences"] < 0:
        raise ValueError("Worker/thread/latency counts must be nonnegative")
    return config


def backend_arguments(config, output):
    require_full(config)
    result = []
    for key, value in config.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif value is not None:
            result.extend((flag, str(value)))
    return result + ["--output", str(Path(output).resolve())]


def preflight(config):
    required = [Path(config[key]) for key in ("train_file", "test_file")]
    backbone = Path(config["roberta_path"])
    required += [backbone / name for name in ("config.json", "vocab.json", "merges.txt")]
    required += [Path(config["lexicon_dir"]) / name for name in ("positive-words.txt", "negative-words.txt")]
    missing = [str(path) for path in required if not path.is_file()]
    weights = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")
    if not any((backbone / name).is_file() for name in weights):
        missing.append(str(backbone) + "/[RoBERTa model weights]")
    if missing:
        raise FileNotFoundError("Prepare external assets first:\n" + "\n".join(missing))
    if Path(config["train_file"]) == Path(config["test_file"]):
        raise ValueError("Training and test files must be different")


def merged_records(path):
    records = read_json(path)
    if not isinstance(records, list) or not records:
        raise ValueError("Processed data must be a nonempty JSON list: " + str(path))
    groups = {}
    for record in records:
        key = json.dumps([record["token"], record["head"], record["amr_tokens"],
                          sorted(record["amr_edges"], key=lambda edge: json.dumps(edge, sort_keys=True))],
                         ensure_ascii=False, sort_keys=True)
        if key not in groups:
            groups[key] = dict(record, aspects=[])
        aspects = groups[key]["aspects"]
        seen = {(a["from"], a["to"]): a for a in aspects}
        for aspect in record["aspects"]:
            span = (aspect["from"], aspect["to"])
            if span in seen:
                old = seen[span]
                if old["term"] != aspect["term"] or old.get("polarity") != aspect.get("polarity"):
                    raise ValueError("Conflicting duplicate aspect")
            else:
                aspects.append(dict(aspect))
                seen[span] = aspect
    return list(groups.values())


def prediction_identity(folder, test_file):
    expected = merged_records(test_file)
    for name in ("best_acc_predictions.json", "best_f1_predictions.json", "test_predictions.json"):
        predictions = read_json(Path(folder) / name)
        if len(expected) != len(predictions):
            raise ValueError("Prediction sentence count differs from current test data")
        for index, (record, row) in enumerate(zip(expected, predictions)):
            if row["sentence_index"] != index or row["tokens"] != record["token"]:
                raise ValueError("Prediction sentence identity mismatch")
            if len(row["aspects"]) != len(record["aspects"]):
                raise ValueError("Prediction aspect count mismatch")
            for gold, predicted in zip(record["aspects"], row["aspects"]):
                if any(predicted[key] != gold[key] for key in ("from", "to", "term")) or predicted["gold"] != gold["polarity"]:
                    raise ValueError("Prediction aspect identity mismatch")


def verify_run(folder, expected):
    from innovations.v1_relation_gate.suite import verify
    folder = Path(folder)
    good, details = verify(folder, expected)
    if not good:
        return good, details
    try:
        config = details["config"]
        require_full(config)
        if config.get("model_variant") != "dagf_aeor_paper_controls_v1":
            raise ValueError("Not the paper-full checkpoint variant")
        if read_json(folder / "completion.json").get("epochs_completed") != config["epochs"]:
            raise ValueError("Training did not finish every configured epoch")
        prediction_identity(folder, expected["test_file"])
        return True, details
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return False, str(error)


def fingerprints(configs):
    paths = set(ROOT.rglob("*.py")) | set((ROOT / "configs").glob("*.json"))
    for config in configs:
        paths.update(Path(config[key]) for key in ("train_file", "test_file"))
        paths.update(Path(config["roberta_path"]) / name for name in ("config.json", "vocab.json", "merges.txt"))
        paths.update(Path(config["lexicon_dir"]) / name for name in ("positive-words.txt", "negative-words.txt"))
    return {str(path.resolve()): sha256(path) for path in sorted(paths)}
