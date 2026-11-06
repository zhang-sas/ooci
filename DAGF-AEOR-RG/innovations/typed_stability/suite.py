"""Stdlib-only, sequential and resumable typed-stability experiment orchestration.

Only the fixed local train_typed_stability.py entry is launched. A completed run
requires the final completion marker, matching configuration and paired metric
artifacts. A partial or failed attempt is retained and retried in a new folder.
"""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("typed", "ema", "schedule", "ema_schedule", "split_lr", "smooth",
            "relation_gate", "semantic_gate")
DATASETS = {
    "restaurant": dict(folder="Restaurants_corenlp", epochs=50, alpha=.8, beta=.6,
                       kl_weight=1.5, max_length=384, lr=2e-5, weight_decay=.01),
    "laptop": dict(folder="Laptops_corenlp", epochs=30, alpha=.5, beta=.4,
                   kl_weight=1.6, max_length=512, lr=2e-5, weight_decay=.01),
    "twitter": dict(folder="Tweets_corenlp", epochs=30, alpha=.6, beta=.6,
                    kl_weight=1.3, max_length=384, lr=1e-5, weight_decay=.0001),
}
ALIASES = {"res": "restaurant", "lap": "laptop"}


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def dataset_name(value):
    value = ALIASES.get(value, value)
    if value not in DATASETS:
        raise ValueError("Dataset must be restaurant, laptop or twitter")
    return value


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise ValueError("Seed/count must be nonnegative")
    return number


def positive_env(name, default, kind=int):
    value = kind(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be positive and finite")
    return value


def new_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_%s_%s" % (os.getpid(), uuid.uuid4().hex[:6])


def resolved_model_path(root):
    if os.environ.get("ROBERTA_PATH"):
        path = Path(os.environ["ROBERTA_PATH"]).expanduser().resolve()
    elif (root / "roberta/config.json").is_file():
        path = root / "roberta"
    else:
        path = root / "roberta/roberta"
    if not (path / "config.json").is_file() and (path / "roberta/config.json").is_file():
        path = path / "roberta"
    return path.resolve()


def settings_for(dataset, root=None):
    root = Path(root or PROJECT_ROOT).resolve()
    d = DATASETS[dataset]
    lr = positive_env("LEARNING_RATE", d["lr"], float)
    batch = positive_env("BATCH_SIZE", 16)
    latency = nonnegative(os.environ.get("LATENCY_SENTENCES", 100))
    settings = dict(
        dataset=dataset, train_file=str(root / "dataset" / d["folder"] / "train_with_amr.json"),
        test_file=str(root / "dataset" / d["folder"] / "test_with_amr.json"),
        roberta_path=str(resolved_model_path(root)),
        lexicon_dir=str(Path(os.environ.get("LEXICON_DIR", str(root / "opinion-lexicon-English"))).expanduser().resolve()),
        epochs=positive_env("EPOCHS", d["epochs"]), batch_size=batch,
        eval_batch_size=positive_env("EVAL_BATCH_SIZE", batch),
        grad_accum_steps=positive_env("GRAD_ACCUM_STEPS", 1), lr=lr,
        weight_decay=d["weight_decay"], dropout=.3, lstm_hidden=300, num_layers=1,
        attention_heads=8, alpha=d["alpha"], beta=d["beta"], kl_weight=d["kl_weight"],
        max_length=d["max_length"], max_amr_edges=2048,
        log_step=positive_env("LOG_STEP", 100), baseline_regularization_weight=0.,
        readout="cls", evaluation_protocol="baseline", device=os.environ.get("DEVICE", "cuda"),
        depth_mode="last", depth_layers=4, depth_selector_dim=64, syntax_features="both", dependency_dim=32,
        ema_start_ratio=.1, ema_device=os.environ.get("EMA_DEVICE", "same"), warmup_ratio=.1,
        edge_gate_dim=32, edge_gate_strength=.5, max_steps=0,
        latency_sentences=latency, latency_repeats=positive_env("LATENCY_REPEATS", 3),
        export_routes=os.environ.get("EXPORT_ROUTES", "1") == "1",
        gradient_checkpointing=os.environ.get("GRADIENT_CHECKPOINTING", "0") == "1",
    )
    if settings["ema_device"] not in ("same", "cpu"):
        raise ValueError("EMA_DEVICE must be same or cpu")
    return settings


def profile_config(settings, profile, seed):
    if profile not in PROFILES:
        raise ValueError("Unknown comparison profile: " + str(profile))
    cfg = dict(settings, comparison_profile=profile, seed=nonnegative(seed),
               added_lr=settings["lr"], ema_decay=0., lr_schedule="none", label_smoothing=0., edge_gate="none")
    if profile in ("ema", "ema_schedule"):
        cfg["ema_decay"] = .99
    if profile in ("schedule", "ema_schedule"):
        cfg["lr_schedule"] = "linear"
    if profile == "split_lr":
        cfg["lr"] *= .5
    if profile == "smooth":
        cfg["label_smoothing"] = .05
    if profile in ("relation_gate", "semantic_gate"):
        cfg["edge_gate"] = profile.split("_")[0]
    return cfg


def preflight(settings, root=None):
    root = Path(root or PROJECT_ROOT)
    required = [Path(settings["roberta_path"]) / "config.json", Path(settings["train_file"]),
                Path(settings["test_file"]), Path(settings["lexicon_dir"]) / "positive-words.txt",
                Path(settings["lexicon_dir"]) / "negative-words.txt", Path(root) / "train_typed_stability.py"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required file missing: " + "\n".join(missing))


def train_command(config, output, root=None):
    root = Path(root or PROJECT_ROOT)
    command = [sys.executable, "-u", "-B", str(Path(root) / "train_typed_stability.py")]
    for key, value in config.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend((flag, str(value)))
    command.extend(("--output", str(Path(output).resolve())))
    return command


def equal_setting(actual, expected):
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (float, int)):
        return not isinstance(actual, bool) and isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15)
    return actual == expected


def metrics_from_predictions(predictions):
    if not isinstance(predictions, list):
        raise ValueError("Predictions must be sentence records")
    cm = [[0]*3 for _ in range(3)]
    seen = set()
    for sentence in predictions:
        for index, a in enumerate(sentence["aspects"]):
            key = (sentence["sentence_index"], a.get("aspect_index", index), a["from"], a["to"])
            if key in seen:
                raise ValueError("Duplicate aspect prediction")
            seen.add(key)
            gold, pred = a["gold_id"], a["pred_id"]
            if isinstance(gold, bool) or isinstance(pred, bool) or gold not in (0,1,2) or pred not in (0,1,2):
                raise ValueError("Unknown polarity in completed predictions")
            cm[gold][pred] += 1
    count = sum(map(sum, cm))
    if not count:
        raise ValueError("Empty prediction export")
    f1 = [2*cm[i][i]/(sum(cm[i])+sum(row[i] for row in cm))
          if sum(cm[i])+sum(row[i] for row in cm) else 0 for i in range(3)]
    return dict(count=count, accuracy=sum(cm[i][i] for i in range(3))/count,
                macro_f1=sum(f1)/3, neutral_f1=f1[2], confusion_matrix=cm)


def verified_branch(output, branch):
    folder = output if branch == "raw" else output / "ema"
    scores = read_json(folder / "best_scores.json")
    if scores.get("evaluation_protocol") != "baseline" or scores.get("selection_split") != "test":
        raise ValueError("Completed scores must use the baseline fulltrain/test protocol")
    result = {}
    for metric, name in (("accuracy", "best_accuracy"), ("macro_f1", "best_macro_f1")):
        summary = scores[name]
        if summary.get("debug_limited"):
            raise ValueError("Debug-limited results are excluded")
        if summary.get("weights_source") != branch:
            raise ValueError("Selected score belongs to a different weight branch")
        checkpoint = scores.get(name + "_checkpoint", "best_acc.pt" if metric == "accuracy" else "best_f1.pt")
        if Path(checkpoint).name != checkpoint:
            raise ValueError("Checkpoint filename must stay in branch folder")
        if not (folder/checkpoint).is_file() or not (folder/checkpoint).stat().st_size:
            raise ValueError("Missing checkpoint: " + str(folder/checkpoint))
        pred_name = "best_acc_predictions.json" if metric == "accuracy" else "best_f1_predictions.json"
        m = metrics_from_predictions(read_json(folder/pred_name))
        if summary.get("count") != m["count"]:
            raise ValueError("Prediction/score count mismatch")
        for key in ("accuracy", "macro_f1"):
            if not equal_setting(summary[key], m[key]):
                raise ValueError("Prediction/score mismatch: " + str(folder/pred_name))
        if summary.get("confusion_matrix") != m["confusion_matrix"]:
            raise ValueError("Confusion matrix differs from prediction export")
        result[name] = dict(summary, neutral_f1=m["neutral_f1"])
    history_path = folder / "history.jsonl"
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not history or any(h.get("debug_limited") for h in history):
        raise ValueError("Missing history or debug-limited history")
    if any(h.get("weights_source") != branch for h in history):
        raise ValueError("History mixes raw and EMA selections")
    if max(h["accuracy"] for h in history) != result["best_accuracy"]["accuracy"] or max(h["macro_f1"] for h in history) != result["best_macro_f1"]["macro_f1"]:
        raise ValueError("History maxima disagree with selected scores")
    report = read_json(folder/"test_report.json")
    if report.get("debug_limited") or report.get("evaluation_protocol") != "baseline":
        raise ValueError("Missing formal baseline test report")
    if report.get("weights_source") != branch:
        raise ValueError("Test report belongs to a different weight branch")
    exported = metrics_from_predictions(read_json(folder/"test_predictions.json"))
    for key in ("accuracy", "macro_f1"):
        if not equal_setting(report.get(key), exported[key]):
            raise ValueError("Final report differs from its prediction export")
    result.update(history_evaluations=len(history), last_epoch=history[-1]["epoch"])
    return result


def verify_completed(output, expected):
    """Fail closed: a score file by itself is never a completion signal."""
    output = Path(output)
    try:
        completion = read_json(output/"completion.json")
        if completion.get("status") != "complete" or completion.get("debug_limited"):
            raise ValueError("Missing complete/non-debug marker")
        for key in ("dataset", "comparison_profile", "seed"):
            if not equal_setting(completion.get(key), expected[key]):
                raise ValueError("Completion metadata mismatch: " + key)
        cfg = read_json(output/"config.json")
        for key, value in expected.items():
            if not equal_setting(cfg.get(key), value):
                raise ValueError("Run configuration mismatch: " + key)
        if cfg.get("max_steps", 0) != 0 or cfg.get("val_file") is not None:
            raise ValueError("Debug/validation runs cannot complete this suite")
        primary = "ema" if expected["ema_decay"] else "raw"
        if completion.get("primary_branch") != primary:
            raise ValueError("Incorrect primary weight branch")
        branches = {"raw": verified_branch(output, "raw")}
        if expected["ema_decay"]:
            branches["ema"] = verified_branch(output, "ema")
        if any(b["last_epoch"] != expected["epochs"] for b in branches.values()):
            raise ValueError("History does not reach configured epochs")
        return True, {"primary_branch":primary,"branches":branches,"completion":completion}
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        return False, str(exc)


@contextmanager
def suite_lock(folder):
    """Kernel lock is released on interruption; no persistent stale-lock trap."""
    lock = (Path(folder)/".suite.lock").open("a+")
    try:
        if os.name == "nt":
            import msvcrt
            lock.seek(0); lock.write(" "); lock.flush(); lock.seek(0)
            try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError: raise RuntimeError("Another process is running this suite")
        else:
            import fcntl
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError: raise RuntimeError("Another process is running this suite")
        yield
    finally:
        lock.close()


def save_manifest(folder, manifest):
    manifest["updated_at"] = timestamp()
    write_json(Path(folder)/"suite_manifest.json", manifest)


def contained(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or root not in path.parents:
        raise ValueError("Run output must stay inside suite folder")
    return path


def execute_job(config, output, log_path, root=None):
    root = Path(root or PROJECT_ROOT)
    command = train_command(config, output, root)
    if Path(output).exists() or Path(log_path).exists():
        raise FileExistsError("Refusing to overwrite an existing attempt")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    print("Command: " + shlex.join(command), flush=True)
    with Path(log_path).open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line); log.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
        finally:
            process.stdout.close()


def csv_write(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def mean_std(values):
    return (statistics.mean(values), statistics.stdev(values) if len(values)>1 else None) if values else (None,None)


def summarize(folder, output=None):
    folder = Path(folder).resolve()
    destination = Path(output).resolve() if output else folder/"summary"
    manifest = read_json(folder/"suite_manifest.json")
    rows, branch_rows, failures = [], [], []
    for job in manifest["jobs"]:
        run = job.get("output")
        if not run:
            failures.append(dict(profile=job["profile"],seed=job["seed"],status=job["status"],reason="not run")); continue
        path=contained(run,folder)
        expected=profile_config(manifest["settings"],job["profile"],job["seed"])
        valid, verification=verify_completed(path,expected)
        if not valid:
            failures.append(dict(profile=job["profile"],seed=job["seed"],status=job["status"],output=str(path),reason=verification)); continue
        primary=verification["primary_branch"]
        for branch,b in verification["branches"].items():
            acc,f1=b["best_accuracy"],b["best_macro_f1"]
            row=dict(dataset=manifest["dataset"],profile=job["profile"],seed=job["seed"],branch=branch,
                     primary_branch=primary,is_primary=branch==primary,output=str(path if branch=="raw" else path/"ema"),
                     max_acc_pct=acc["accuracy"]*100,max_macro_f1_pct=f1["macro_f1"]*100,
                     acc_at_max_f1_pct=f1["accuracy"]*100,f1_at_max_acc_pct=acc["macro_f1"]*100,
                     neutral_f1_at_max_f1_pct=f1["neutral_f1"]*100,
                     best_acc_epoch=acc["epoch"],best_acc_updates=acc["optimizer_steps"],
                     best_f1_epoch=f1["epoch"],best_f1_updates=f1["optimizer_steps"],
                     evaluation_count=b["history_evaluations"])
            branch_rows.append(row)
            if branch==primary: rows.append(row)
    grouped=[]
    for profile in manifest["profiles"]:
        members=[r for r in rows if r["profile"]==profile]
        row=dict(dataset=manifest["dataset"],profile=profile,n=len(members),expected_runs=len(manifest["seeds"]),completed_runs=len(members),
                 complete_all_seeds=len(members)==len(manifest["seeds"]),seeds=",".join(str(r["seed"]) for r in members))
        for metric in ("max_acc_pct","max_macro_f1_pct","neutral_f1_at_max_f1_pct"):
            mean,std=mean_std([r[metric] for r in members]);row[metric+"_mean"]=mean;row[metric+"_std"]=std
        grouped.append(row)
    reference={r["seed"]:r for r in rows if r["profile"]=="typed"}
    paired=[]
    for row in rows:
        if row["profile"]=="typed" or row["seed"] not in reference: continue
        base=reference[row["seed"]]
        paired.append(dict(dataset=row["dataset"],profile=row["profile"],seed=row["seed"],reference_profile="typed",
                           delta_max_acc_pp=row["max_acc_pct"]-base["max_acc_pct"],
                           delta_max_macro_f1_pp=row["max_macro_f1_pct"]-base["max_macro_f1_pct"],
                           delta_neutral_f1_pp=row["neutral_f1_at_max_f1_pct"]-base["neutral_f1_at_max_f1_pct"]))
    paired_grouped=[]
    for profile in manifest["profiles"]:
        if profile=="typed":continue
        values=[p for p in paired if p["profile"]==profile]
        row=dict(profile=profile,paired_seeds=len(values),expected_seeds=len(manifest["seeds"]))
        for metric in ("delta_max_acc_pp","delta_max_macro_f1_pp","delta_neutral_f1_pp"):
            mean,std=mean_std([v[metric] for v in values]);row[metric+"_mean"]=mean;row[metric+"_std"]=std
        paired_grouped.append(row)
    result=dict(created_at=timestamp(),suite=str(folder),expected_runs=len(manifest["jobs"]),completed_runs=len(rows),
                complete_all_runs=len(rows)==len(manifest["jobs"]),per_run=rows,per_branch=branch_rows,
                grouped=grouped,paired_by_seed=paired,paired_grouped=paired_grouped,failed_or_missing=failures,
                metric_note="All maxima are independently selected per run at fixed periodic test evaluations. Sample standard deviations across seeds; no best-seed substitution. Incomplete seed sets are explicitly flagged. " +
                    ("Single seed only: no claim about training-seed stability; standard deviation is unavailable." if len(manifest["seeds"])==1 else "Multiple explicitly requested training seeds."))
    destination.mkdir(parents=True,exist_ok=True)
    for name,records in (("per_run",rows),("per_branch",branch_rows),("grouped",grouped),("paired_by_seed",paired),("paired_grouped",paired_grouped),("failed_or_missing",failures)):
        write_json(destination/(name+".json"),records)
        fields=list(records[0]) if records else {"per_run":["dataset","profile","seed","max_macro_f1_pct"],
            "per_branch":["dataset","profile","seed","branch","max_macro_f1_pct"],
            "grouped":["dataset","profile","expected_runs","completed_runs","complete_all_seeds"],
            "paired_by_seed":["dataset","profile","seed","delta_max_macro_f1_pp"],
            "paired_grouped":["profile","paired_seeds","expected_seeds"],
            "failed_or_missing":["profile","seed","status","reason"]}[name]
        # Failure rows may include output only after an attempt exists.
        fields=list(dict.fromkeys(fields+[k for r in records for k in r]))
        csv_write(destination/(name+".csv"),records,fields)
    write_json(destination/"summary.json",result)
    return result


def make_manifest(dataset, profiles, seeds, settings):
    return dict(schema_version=1,created_at=timestamp(),dataset=dataset,profiles=list(profiles),seeds=list(seeds),
                settings=settings,host=socket.gethostname(),status="planned",
                jobs=[dict(profile=p,seed=s,status="pending",output=None,attempts=[]) for s in seeds for p in profiles])


def run_suite(args):
    if args.resume:
        folder=args.resume.resolve()
        manifest=read_json(folder/"suite_manifest.json")
        if manifest.get("schema_version")!=1: raise ValueError("Unsupported suite manifest version")
        if args.dataset and dataset_name(args.dataset)!=manifest["dataset"]: raise ValueError("Resume dataset differs from saved suite")
        if args.selection and list(PROFILES if args.selection=="all" else PROFILES[:4])!=manifest["profiles"]: raise ValueError("Resume profile set differs from saved suite")
        if args.seeds is not None and args.seeds!=manifest["seeds"]: raise ValueError("Resume seeds differ from saved suite")
    else:
        dataset=dataset_name(args.dataset or "restaurant")
        profiles=PROFILES if (args.selection or "all")=="all" else PROFILES[:4]
        seeds=args.seeds if args.seeds is not None else [nonnegative(x) for x in os.environ.get("SEEDS","1000").replace(","," ").split()]
        if not seeds or len(set(seeds))!=len(seeds):raise ValueError("Seeds must be nonempty and unique")
        manifest=make_manifest(dataset,profiles,seeds,settings_for(dataset))
        parent=args.output_root or Path(os.environ.get("OUTPUT_ROOT",str(PROJECT_ROOT/"runs_typed_stability")))
        folder=parent.resolve()/"suites"/(dataset+"_"+new_id())
    preflight(manifest["settings"])
    if args.dry_run or os.environ.get("DRY_RUN")=="1":
        print("Dry run: %s profiles x %s seeds = %s trainings; output %s" % (len(manifest["profiles"]),len(manifest["seeds"]),len(manifest["jobs"]),folder))
        for job in manifest["jobs"]:
            output=folder/"jobs"/job["profile"]/("seed_"+str(job["seed"]))/"attempt_0001"
            print(shlex.join(train_command(profile_config(manifest["settings"],job["profile"],job["seed"]),output)))
        return 0
    if not args.resume:
        folder.mkdir(parents=True,exist_ok=False)
        save_manifest(folder,manifest)
    print("Suite: "+str(folder),flush=True)
    exit_code=0
    with suite_lock(folder):
        manifest["status"]="running";save_manifest(folder,manifest)
        try:
            for number,job in enumerate(manifest["jobs"],1):
                cfg=profile_config(manifest["settings"],job["profile"],job["seed"])
                if job.get("output"):
                    previous=contained(job["output"],folder)
                    good,detail=verify_completed(previous,cfg)
                    if good:
                        job["status"]="complete";job["verification"]="verified_complete";save_manifest(folder,manifest)
                        print("[%s/%s] Skip verified complete %s seed %s" % (number,len(manifest["jobs"]),job["profile"],job["seed"]),flush=True)
                        continue
                    job["previous_verification_error"]=detail
                parent=folder/"jobs"/job["profile"]/("seed_"+str(job["seed"]))
                attempt=len(job["attempts"])+1
                output=parent/("attempt_%04d" % attempt)
                while output.exists() or output.with_suffix(".log").exists():
                    attempt+=1;output=parent/("attempt_%04d" % attempt)
                output=contained(output,folder)
                log=output.with_suffix(".log")
                record=dict(output=str(output),log=str(log),started_at=timestamp(),status="running")
                job["attempts"].append(record);job.update(output=str(output),status="running")
                save_manifest(folder,manifest)
                print("[%s/%s] %s seed %s" % (number,len(manifest["jobs"]),job["profile"],job["seed"]),flush=True)
                try:
                    code=execute_job(cfg,output,log)
                    good,detail=verify_completed(output,cfg) if code==0 else (False,"Training process exited %s" % code)
                    record.update(exit_code=code,status="complete" if good else "failed",finished_at=timestamp())
                    job["status"]=record["status"]
                    if not good:
                        record["error"]=detail;job["error"]=detail;exit_code=1
                        print("FAILED: "+str(detail),file=sys.stderr,flush=True)
                    else: job.pop("error",None)
                except (OSError,ValueError) as exc:
                    record.update(status="failed",error=str(exc),finished_at=timestamp())
                    job.update(status="failed",error=str(exc));exit_code=1
                    print("FAILED: "+str(exc),file=sys.stderr,flush=True)
                save_manifest(folder,manifest)
        except BaseException:
            manifest["status"]="interrupted";save_manifest(folder,manifest)
            summarize(folder)
            raise
        result=summarize(folder)
        manifest["status"]="complete" if result["complete_all_runs"] else "finished_with_failures"
        save_manifest(folder,manifest)
        if not result["complete_all_runs"]:exit_code=1
    print("Summary: "+str(folder/"summary/grouped.csv"),flush=True)
    print("Verified %s/%s runs. Resume: bash run_typed_stability_suite.sh --resume %s" % (result["completed_runs"],result["expected_runs"],shlex.quote(str(folder))),flush=True)
    return exit_code


def single(args):
    dataset=dataset_name(args.dataset)
    settings=settings_for(dataset);preflight(settings)
    cfg=profile_config(settings,args.profile,args.seed)
    parent=args.output_root or Path(os.environ.get("OUTPUT_ROOT",str(PROJECT_ROOT/"runs_typed_stability")))
    output=parent.resolve()/dataset/args.profile/("seed_"+str(args.seed))/new_id()
    if args.dry_run or os.environ.get("DRY_RUN")=="1":
        print(shlex.join(train_command(cfg,output)));return 0
    code=execute_job(cfg,output,output.with_suffix(".log"))
    if code:return code
    good,detail=verify_completed(output,cfg)
    if not good:
        print("Training exited but completion verification failed: "+str(detail),file=sys.stderr);return 1
    print("Finished and verified: "+str(output));return 0


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="mode",required=True)
    one=sub.add_parser("single",help="One independent profile/seed")
    one.add_argument("dataset",nargs="?",default="restaurant")
    one.add_argument("profile",nargs="?",default="typed",choices=PROFILES)
    one.add_argument("seed",nargs="?",default=1000,type=nonnegative)
    one.add_argument("--output-root",type=Path);one.add_argument("--dry-run",action="store_true")
    suite=sub.add_parser("suite",help="All eight profiles (or core four), sequentially")
    suite.add_argument("dataset",nargs="?")
    suite.add_argument("selection",nargs="?",choices=("all","core"))
    suite.add_argument("--seeds",type=nonnegative,nargs="+")
    suite.add_argument("--resume",type=Path)
    suite.add_argument("--output-root",type=Path);suite.add_argument("--dry-run",action="store_true")
    return parser


def summary_main(argv=None):
    parser=argparse.ArgumentParser(description="Summarize verified typed-stability suite results")
    parser.add_argument("--suite",type=Path,required=True);parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    result=summarize(args.suite,args.output)
    print(json.dumps({k:result[k] for k in ("suite","expected_runs","completed_runs","complete_all_runs")},ensure_ascii=False))


def main(argv=None):
    args=build_parser().parse_args(argv)
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Experiment runner interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    try:return single(args) if args.mode=="single" else run_suite(args)
    except (OSError,ValueError,RuntimeError) as exc:
        print("Error: "+str(exc),file=sys.stderr);return 2
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__=="__main__":
    raise SystemExit(main())
