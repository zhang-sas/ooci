"""Run full-model seeds sequentially and report paired metrics and sample SD."""
import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import uuid

sys.dont_write_bytecode = True
from aeor_rg.runtime import (ROOT, PAPER_DATASETS, canonical_dataset, config_for, preflight,
                            read_json, write_json, fingerprints, verify_run)
from innovations.typed_stability.suite import suite_lock

SCHEMA = "aeor_rg_public_full_seeds_v1"
DEFAULT_SEEDS = [1000, 1001, 1002, 1003, 1004]
OVERRIDES = ("device", "roberta_path", "lexicon_dir", "epochs", "log_step", "batch_size", "eval_batch_size")
METRICS = ("max_acc_pct", "max_macro_f1_pct", "acc_at_best_f1_pct")


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(folder):
    folder = Path(folder)
    manifest = read_json(folder / "suite_manifest.json")
    rows = []
    for job in manifest["jobs"]:
        good, detail = verify_run(job["output"], job["config"]) if job.get("output") else (False, "Not run")
        row = dict(dataset=job["dataset"], seed=job["seed"], status="complete" if good else "incomplete",
                   output=job.get("output"), epochs=job["config"]["epochs"], log_step=job["config"]["log_step"])
        if good:
            best_acc, best_f1 = detail["scores"]["best_accuracy"], detail["scores"]["best_macro_f1"]
            row.update(max_acc_pct=100 * best_acc["accuracy"], max_macro_f1_pct=100 * best_f1["macro_f1"],
                       acc_at_best_f1_pct=100 * best_f1["accuracy"], best_f1_epoch=best_f1["epoch"],
                       best_f1_updates=best_f1["optimizer_steps"], evaluation_count=detail["evaluation_count"])
        else:
            row["reason"] = str(detail)
        rows.append(row)
    groups = []
    for dataset in manifest["datasets"]:
        complete = [row for row in rows if row["dataset"] == dataset and row["status"] == "complete"]
        ready = len(complete) == len(manifest["seeds"]) and {row["seed"] for row in complete} == set(manifest["seeds"])
        group = dict(dataset=dataset, expected_runs=len(manifest["seeds"]), completed_runs=len(complete), complete_all_seeds=ready)
        for metric in METRICS:
            values = [row[metric] for row in complete]
            group[metric + "_mean"] = statistics.mean(values) if ready else None
            group[metric + "_std"] = statistics.stdev(values) if ready and len(values) > 1 else None
        groups.append(group)
    result = dict(schema=SCHEMA, complete_all_runs=all(group["complete_all_seeds"] for group in groups),
                  datasets=manifest["datasets"], seeds=manifest["seeds"], rows=rows, grouped=groups,
                  units="Acc/F1 in percent; sample SD (ddof=1) in percentage points",
                  selection_protocol="Full train; periodic test and epoch-end test; independent Acc/F1 maxima",
                  missing_seed_policy="No mean/SD until all requested seeds complete")
    write_json(folder / "summary.json", result)
    csv_write(folder / "per_seed.csv", rows)
    csv_write(folder / "mean_std.csv", groups)
    return result


def command(job, output, overrides):
    result = [sys.executable, "-u", "-B", str(ROOT / "train.py"), "--dataset", job["dataset"],
              "--seed", str(job["seed"]), "--output", str(output)]
    for key, value in overrides.items():
        if value is not None:
            result.extend(("--" + key.replace("_", "-"), str(value)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="all", choices=("all", "restaurant", "res", "laptop", "lap", "mams", "twitter"))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs_suites")
    parser.add_argument("--device")
    parser.add_argument("--roberta-path", type=Path)
    parser.add_argument("--lexicon-dir", type=Path)
    for key in ("epochs", "log-step", "batch-size", "eval-batch-size"):
        parser.add_argument("--" + key, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--summarize", type=Path)
    args = parser.parse_args(argv)
    if args.summarize:
        result = summarize(args.summarize.resolve())
        print(json.dumps(result["grouped"], ensure_ascii=False, indent=2))
        return 0 if result["complete_all_runs"] else 1
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Seeds must be distinct nonnegative integers")
    name = canonical_dataset(args.dataset)
    datasets = list(PAPER_DATASETS) if name == "all" else [name]
    overrides = {key: getattr(args, key) for key in OVERRIDES}
    jobs = [dict(id=dataset + "/seed_" + str(seed), dataset=dataset, seed=seed,
                 config=config_for(dataset, seed, overrides), output=None, attempts=[], status="pending")
            for dataset in datasets for seed in args.seeds]
    folder = args.output_root.resolve() / ("full_" + name + "_seeds_" + "_".join(map(str, args.seeds)))
    if args.fresh:
        folder = folder.with_name(folder.name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
    if args.dry_run:
        print("FULL MODEL ONLY: %d datasets x %d seeds = %d independent runs" % (len(datasets), len(args.seeds), len(jobs)))
        for job in jobs:
            print(shlex.join(command(job, folder / job["id"] / "attempt_0001", overrides)))
        print("No model loading, training, inference or run-directory creation.")
        return 0
    for job in jobs:
        preflight(job["config"])
    snapshot = fingerprints([job["config"] for job in jobs])
    folder.mkdir(parents=True, exist_ok=True)
    with suite_lock(folder):
        target = folder / "suite_manifest.json"
        if target.is_file():
            manifest = read_json(target)
            if (manifest.get("schema") != SCHEMA or manifest.get("fingerprints") != snapshot
                    or [(j["id"], j["config"]) for j in manifest["jobs"]] != [(j["id"], j["config"]) for j in jobs]):
                raise ValueError("Inputs/code/configuration changed; use --fresh or another --output-root")
        else:
            manifest = dict(schema=SCHEMA, datasets=datasets, seeds=args.seeds, jobs=jobs, fingerprints=snapshot, status="running")
        write_json(target, manifest)
        try:
            for index, job in enumerate(manifest["jobs"], 1):
                if job.get("output") and verify_run(job["output"], job["config"])[0]:
                    job["status"] = "complete"
                    print("Verified complete, skip: " + job["id"], flush=True)
                    continue
                output = folder / job["id"] / ("attempt_%04d" % (len(job["attempts"]) + 1))
                while output.exists():
                    job["attempts"].append(str(output))
                    output = folder / job["id"] / ("attempt_%04d" % (len(job["attempts"]) + 1))
                output.parent.mkdir(parents=True, exist_ok=True)
                job.update(output=str(output), status="running")
                job["attempts"].append(str(output))
                write_json(target, manifest)
                print("[%d/%d] %s -> %s" % (index, len(jobs), job["id"], output), flush=True)
                environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
                                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                with output.with_suffix(".log").open("w", encoding="utf-8") as log:
                    with subprocess.Popen(command(job, output, overrides), cwd=str(ROOT), env=environment,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                          encoding="utf-8", errors="replace", bufsize=1) as process:
                        for line in process.stdout:
                            print(line, end="", flush=True)
                            log.write(line)
                            log.flush()
                        code = process.wait()
                good, detail = verify_run(output, job["config"]) if code == 0 else (False, "Process exit %d" % code)
                job.update(status="complete" if good else "failed", error=None if good else str(detail))
                write_json(target, manifest)
                summarize(folder)
        except BaseException:
            manifest["status"] = "interrupted"
            write_json(target, manifest)
            raise
        result = summarize(folder)
        manifest["status"] = "complete" if result["complete_all_runs"] else "finished_with_failures"
        write_json(target, manifest)
    print("Per seed: " + str(folder / "per_seed.csv"))
    print("Mean/SD: " + str(folder / "mean_std.csv"))
    return 0 if result["complete_all_runs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
