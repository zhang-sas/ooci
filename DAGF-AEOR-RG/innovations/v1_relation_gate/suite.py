"""Two pure-v1 profiles, periodic step evaluation and independently checked summaries."""
import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

from ..typed_stability.suite import (settings_for,write_json,read_json,metrics_from_predictions,
                                   equal_setting,suite_lock,contained)

ROOT=Path(__file__).resolve().parents[2]
PROFILES=("v1","v1_relation_gate")
DROP_FIELDS={"depth_mode","depth_layers","depth_selector_dim","syntax_features","dependency_dim",
             "ema_start_ratio","ema_device","warmup_ratio","edge_gate_dim","edge_gate_strength","log_step"}


def make_config(dataset,profile,seed,interval):
    # MAMS is an explicit engineering preset, not an original-paper setting.
    # Reuse shared path/runtime options without modifying older suite presets.
    settings=settings_for("restaurant" if dataset=="mams" else dataset,root=ROOT)
    if dataset=="mams":
        settings.update(dataset="mams",train_file=str(ROOT/"dataset/MAMS_corenlp/train_with_amr.json"),
            test_file=str(ROOT/"dataset/MAMS_corenlp/test_with_amr.json"),
            epochs=int(os.environ.get("EPOCHS","30")),alpha=.8,beta=.6,kl_weight=1.5,max_length=384)
        if settings["epochs"]<1:raise ValueError("EPOCHS must be positive")
    config={k:v for k,v in settings.items() if k not in DROP_FIELDS}
    config.update(comparison_profile=profile,seed=seed,added_lr=config["lr"],log_step=interval,
        relation_reweight=profile=="v1_relation_gate",gate_relation_dim=32,gate_hidden_dim=32,gate_strength=.5)
    return config


def command(config,output):
    cmd=[sys.executable,"-u","-B",str(ROOT/"train_v1_relation_gate.py")]
    for key,value in config.items():
        if isinstance(value,bool):
            if value:cmd.append("--"+key.replace("_","-"))
        else:cmd.extend(("--"+key.replace("_","-"),str(value)))
    return cmd+["--output",str(output)]


def preflight(config):
    paths=[Path(config["train_file"]),Path(config["test_file"]),Path(config["roberta_path"])/"config.json",
        Path(config["lexicon_dir"])/"positive-words.txt",Path(config["lexicon_dir"])/"negative-words.txt",
        ROOT/"train_v1_relation_gate.py"]
    missing=[str(p) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError("Required file missing: "+", ".join(missing))


def expected_evaluations(sentences,batch_size,epochs,log_step,grad_accum_steps=1):
    per_epoch=math.ceil(math.ceil(sentences/batch_size)/grad_accum_steps)
    if per_epoch<1:raise ValueError("Training dataset must be nonempty")
    points=[]
    for epoch in range(1,epochs+1):
        start,end=(epoch-1)*per_epoch,epoch*per_epoch
        points.extend((epoch,step) for step in range((start//log_step+1)*log_step,end+1,log_step))
        if end%log_step:points.append((epoch,end))
    return points


def verify(output,expected):
    try:
        cfg=read_json(output/"config.json");done=read_json(output/"completion.json")
        if done.get("status")!="complete" or done.get("debug_limited"):raise ValueError("No formal completion marker")
        for k,v in expected.items():
            if not equal_setting(cfg.get(k),v):raise ValueError("Config mismatch: "+k)
        if cfg.get("log_step",0)<1 or cfg.get("additive_relation_messages") is not False:
            raise ValueError("Expected pure-v1 configuration with positive step interval")
        if cfg.get("eval_interval_unit")!="optimizer_step" or cfg.get("val_file") is not None:
            raise ValueError("Expected fulltrain/periodic-test protocol with optimizer-step interval")
        history=[__import__("json").loads(x) for x in (output/"history.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        data=read_json(output/"data_summary.json")
        points=expected_evaluations(data["train"]["sentences"],cfg["batch_size"],cfg["epochs"],cfg["log_step"],cfg["grad_accum_steps"])
        if [(h["epoch"],h["optimizer_steps"]) for h in history]!=points:
            raise ValueError("Expected periodic optimizer-step tests plus epoch-end tests")
        if any(h.get("debug_limited") for h in history):raise ValueError("Debug results cannot be compared")
        scores=read_json(output/"best_scores.json")
        if scores.get("evaluation_protocol")!="baseline" or scores.get("selection_split")!="test":raise ValueError("Wrong selection protocol")
        computed={}
        for name,key in (("best_acc","best_accuracy"),("best_f1","best_macro_f1")):
            if not (output/(name+".pt")).is_file():raise ValueError("Missing "+name+" checkpoint")
            m=metrics_from_predictions(read_json(output/(name+"_predictions.json")))
            if any(abs(m[k]-scores[key][k])>1e-12 for k in ("accuracy","macro_f1")) or m["confusion_matrix"]!=scores[key]["confusion_matrix"]:
                raise ValueError("Prediction/score mismatch: "+name)
            computed[name]=m
        if max(h["accuracy"] for h in history)!=scores["best_accuracy"]["accuracy"] or max(h["macro_f1"] for h in history)!=scores["best_macro_f1"]["macro_f1"]:
            raise ValueError("History maximum differs from saved scores")
        report=read_json(output/"test_report.json")
        if report.get("debug_limited") or report.get("selected_checkpoint")!="best_acc.pt":raise ValueError("Missing final baseline report")
        final=metrics_from_predictions(read_json(output/"test_predictions.json"))
        if abs(final["macro_f1"]-report["macro_f1"])>1e-12:raise ValueError("Final prediction mismatch")
        return True,dict(scores=scores,computed=computed,evaluation_count=len(history),config=cfg)
    except (OSError,ValueError,TypeError,KeyError,IndexError) as exc:return False,str(exc)


def summarize(folder):
    import csv
    folder=Path(folder).resolve();manifest=read_json(folder/"suite_manifest.json");rows=[]
    for job in manifest["jobs"]:
        good,detail=verify(contained(job["output"],folder),job["config"]) if job.get("output") else (False,"Not run")
        if not good:
            rows.append(dict(profile=job["profile"],status="failed_or_incomplete",reason=detail));continue
        best=detail["scores"]["best_macro_f1"];m=detail["computed"]
        rows.append(dict(profile=job["profile"],status="complete",seed=manifest["seed"],
            evaluation_unit="optimizer_step",log_step=manifest["interval"],evaluation_count=detail["evaluation_count"],
            max_acc_pct=m["best_acc"]["accuracy"]*100,max_macro_f1_pct=m["best_f1"]["macro_f1"]*100,
            neutral_f1_pct=m["best_f1"]["neutral_f1"]*100,best_f1_epoch=best["epoch"],
            best_f1_steps=best["optimizer_steps"],output=job["output"]))
    base=next((r for r in rows if r["profile"]=="v1" and r["status"]=="complete"),None)
    for row in rows:
        if base and row["status"]=="complete":row["delta_same_round_v1_f1_pp"]=row["max_macro_f1_pct"]-base["max_macro_f1_pct"]
    result=dict(seed=manifest["seed"],evaluation_unit="optimizer_step",log_step=manifest["interval"],
        complete_all_runs=all(r["status"]=="complete" for r in rows),rows=rows,
        note="Pure frozen v1 versus pure v1 with edge-only relation gates; no typed additive relation message. Best scores use periodic optimizer-step and epoch-end tests.")
    write_json(folder/"summary.json",result)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (folder/"summary.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return result


def launch(config,output):
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.with_suffix(".log").open("x",encoding="utf-8") as log:
        process=subprocess.Popen(command(config,output),cwd=str(ROOT),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding="utf-8",errors="replace")
        try:
            for line in process.stdout:
                log.write(line);log.flush();sys.stdout.write(line);sys.stdout.flush()
            return process.wait()
        except BaseException:
            process.terminate();process.wait();raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset",nargs="?",default="restaurant",choices=("restaurant","laptop","twitter","mams"))
    parser.add_argument("mode",nargs="?",default="compare",choices=("compare","v1","gate"))
    parser.add_argument("seed",nargs="?",type=int,default=1000)
    parser.add_argument("--log-step",type=int,default=20)
    parser.add_argument("--output-root",type=Path)
    parser.add_argument("--resume",type=Path)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args(argv)
    if args.seed<0 or args.log_step<1:parser.error("Seed nonnegative; optimizer-step interval positive")
    if args.resume:
        folder=args.resume.resolve();manifest=read_json(folder/"suite_manifest.json")
        if manifest.get("schema")!="pure_v1_relation_gate_steps":raise ValueError("Wrong comparison manifest")
    else:
        profiles=PROFILES if args.mode=="compare" else ("v1",) if args.mode=="v1" else ("v1_relation_gate",)
        parent=args.output_root or Path(os.environ.get("OUTPUT_ROOT",str(ROOT/"runs_v1_relation_gate")))
        folder=parent.resolve()/"suites"/(args.dataset+"_"+datetime.now().strftime("%Y%m%d_%H%M%S")+"_%s_"%os.getpid()+uuid.uuid4().hex[:6])
        manifest=dict(schema="pure_v1_relation_gate_steps",seed=args.seed,interval=args.log_step,status="planned",
            jobs=[dict(profile=p,config=make_config(args.dataset,p,args.seed,args.log_step),output=None,status="pending",attempts=[]) for p in profiles])
    for job in manifest["jobs"]:preflight(job["config"])
    if args.dry_run or os.environ.get("DRY_RUN")=="1":
        print("Dry run: %s training(s), seed %s; test every %s optimizer STEPS plus epoch ends"%(len(manifest["jobs"]),manifest["seed"],manifest["interval"]))
        for job in manifest["jobs"]:
            print(shlex.join(command(job["config"],folder/job["profile"]/"attempt_0001")))
        return 0
    if not args.resume:folder.mkdir(parents=True,exist_ok=False)
    print("Comparison: "+str(folder),flush=True)
    with suite_lock(folder):
        manifest["status"]="running";write_json(folder/"suite_manifest.json",manifest)
        try:
            for job in manifest["jobs"]:
                if job.get("output") and verify(contained(job["output"],folder),job["config"])[0]:
                    job["status"]="complete";continue
                attempt=len(job["attempts"])+1
                output=folder/job["profile"]/("attempt_%04d"%attempt)
                while output.exists() or output.with_suffix(".log").exists():
                    attempt+=1;output=folder/job["profile"]/("attempt_%04d"%attempt)
                job.update(output=str(output),status="running");job["attempts"].append(str(output))
                write_json(folder/"suite_manifest.json",manifest)
                try:
                    code=launch(job["config"],output)
                    good,detail=verify(output,job["config"]) if code==0 else (False,"Process exit %s"%code)
                    job.update(status="complete" if good else "failed",error=None if good else detail)
                except (OSError,ValueError) as exc:job.update(status="failed",error=str(exc))
                write_json(folder/"suite_manifest.json",manifest)
        except BaseException:
            manifest["status"]="interrupted";write_json(folder/"suite_manifest.json",manifest);raise
        result=summarize(folder)
        manifest["status"]="complete" if result["complete_all_runs"] else "finished_with_failures"
        write_json(folder/"suite_manifest.json",manifest)
    print("Summary: "+str(folder/"summary.csv"),flush=True)
    return 0 if result["complete_all_runs"] else 1


def summary_main(argv=None):
    parser=argparse.ArgumentParser(description="Recompute the pure-v1 comparison summary")
    parser.add_argument("--root",type=Path,required=True)
    args=parser.parse_args(argv)
    print(__import__("json").dumps(summarize(args.root),ensure_ascii=False,indent=2))


if __name__=="__main__":raise SystemExit(main())
