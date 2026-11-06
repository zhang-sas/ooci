"""Train only the full AEOR + RG model using the unchanged training backend."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import uuid

sys.dont_write_bytecode = True
from aeor_rg.runtime import ROOT, config_for, backend_arguments, preflight

OVERRIDES = ("device", "train_file", "test_file", "roberta_path", "lexicon_dir", "epochs", "log_step",
             "batch_size", "eval_batch_size", "lr", "added_lr", "weight_decay", "grad_accum_steps",
             "max_length", "max_amr_edges", "num_workers", "threads", "latency_sentences", "latency_repeats",
             "export_routes", "gradient_checkpointing")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("restaurant", "res", "laptop", "lap", "mams", "twitter"), default="restaurant")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    for key in ("train-file", "test-file", "roberta-path", "lexicon-dir"):
        parser.add_argument("--" + key, type=Path)
    for key in ("epochs", "log-step", "batch-size", "eval-batch-size", "grad-accum-steps", "max-length",
                "max-amr-edges", "num-workers", "threads", "latency-sentences", "latency-repeats"):
        parser.add_argument("--" + key, type=int)
    for key in ("lr", "added-lr", "weight-decay"):
        parser.add_argument("--" + key, type=float)
    parser.add_argument("--export-routes", action="store_true", default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print settings without importing torch or creating a run")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    overrides = {key: getattr(args, key) for key in OVERRIDES}
    if args.lr is not None and args.added_lr is None:
        overrides["added_lr"] = args.lr
    config = config_for(args.dataset, args.seed, overrides)
    output = args.output or ROOT / "runs" / config["dataset"] / ("seed_" + str(args.seed)) / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    if args.dry_run:
        print(json.dumps(dict(config=config, output=str(output.resolve()), training_performed=False), ensure_ascii=False, indent=2))
        return 0
    preflight(config)
    from innovations.paper_experiments.train import main as train_full
    train_full(backend_arguments(config, output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
