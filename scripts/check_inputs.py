"""Check processed input structure without importing torch or running a model."""
import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aeor_rg.runtime import PAPER_DATASETS, canonical_dataset, config_for, preflight, read_json, merged_records


def check_data(path):
    records = read_json(path)
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty JSON list: " + str(path))
    for index, record in enumerate(records):
        where = "%s record %d" % (path, index)
        for key in ("token", "head", "deprel", "amr_tokens", "amr_edges", "aspects"):
            if key not in record or not isinstance(record[key], list):
                raise ValueError(where + ": missing/list field " + key)
        words = record["token"]
        if not words or any(not isinstance(word, str) or not word.strip() for word in words):
            raise ValueError(where + ": tokens must be nonblank strings")
        if len(record["head"]) != len(words) or len(record["deprel"]) != len(words):
            raise ValueError(where + ": dependency lengths do not match token length")
        if any(isinstance(head, bool) or not isinstance(head, int) or not 0 <= head <= len(words) for head in record["head"]):
            raise ValueError(where + ": heads must use 1-based indices with root=0")
        if any(not isinstance(label, str) or not label.strip() for label in record["deprel"]):
            raise ValueError(where + ": invalid dependency relation labels")
        amr_words = record["amr_tokens"]
        if not amr_words or any(not isinstance(word, str) or not word.strip() for word in amr_words):
            raise ValueError(where + ": invalid AMR tokens")
        normalize = lambda tokens: "".join(c for word in tokens for c in word.casefold() if not c.isspace())
        if normalize(words) != normalize(amr_words):
            raise ValueError(where + ": AMR must be aligned to the same text; concept-node indices are unsupported")
        for edge in record["amr_edges"]:
            if (not isinstance(edge, (list, tuple)) or len(edge) != 3
                    or not isinstance(edge[1], str) or not edge[1].strip()
                    or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < len(amr_words) for v in (edge[0], edge[2]))):
                raise ValueError(where + ": invalid [source, relation, target] AMR edge")
        if not record["aspects"]:
            raise ValueError(where + ": no labelled aspects")
        for aspect in record["aspects"]:
            start, end = aspect.get("from"), aspect.get("to")
            if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int)
                    or not isinstance(end, int) or not 0 <= start < end <= len(words)):
                raise ValueError(where + ": invalid [from,to) aspect span")
            if not isinstance(aspect.get("term"), list) or aspect["term"] != words[start:end]:
                raise ValueError(where + ": aspect term/token span mismatch")
            if aspect.get("polarity") not in ("positive", "negative", "neutral"):
                raise ValueError(where + ": expected positive/negative/neutral aspect polarity")
    merged = merged_records(path)
    return dict(path=str(path), raw_records=len(records), merged_sentences=len(merged),
                merged_aspects=sum(len(record["aspects"]) for record in merged))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="all", choices=("all", "restaurant", "res", "laptop", "lap", "mams", "twitter"))
    parser.add_argument("--file", type=Path, help="Check one processed JSON file instead of configured datasets")
    parser.add_argument("--data-only", action="store_true", help="Skip backbone/lexicon existence checks")
    parser.add_argument("--roberta-path", type=Path)
    parser.add_argument("--lexicon-dir", type=Path)
    args = parser.parse_args(argv)
    if args.file:
        print(json.dumps(check_data(args.file), ensure_ascii=False, indent=2))
        return 0
    datasets = PAPER_DATASETS if args.dataset == "all" else [canonical_dataset(args.dataset)]
    for dataset in datasets:
        config = config_for(dataset, overrides=dict(roberta_path=args.roberta_path, lexicon_dir=args.lexicon_dir))
        if not args.data_only:
            preflight(config)
        print(json.dumps(dict(dataset=dataset, train=check_data(Path(config["train_file"])),
                              test=check_data(Path(config["test_file"]))), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
