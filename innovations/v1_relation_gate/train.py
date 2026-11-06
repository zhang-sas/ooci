"""Independent pure-v1 comparison with periodic optimizer-step and epoch-end tests."""
import argparse
import json
from pathlib import Path
import random
import uuid
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast
from . import VARIANT
from .model import V1RelationGateDAGF
from .evaluation import evaluate,benchmark_latency,model_inputs,evaluate_counterfactual,_dataset_class
from ..v1_reference.data import DATASET_FOLDERS,build_amr_vocab
from ..v1_reference.model import sentence_cross_entropy
from ..typed_syntax.data import build_dependency_vocab,validate_dependency_vocab,typed_collate as sentence_collate
from ..target_readout import train as common
from ..target_readout.train import PROJECT_ROOT,resolve_defaults,_check_device,write_json,save_checkpoint
from ..typed_stability.checkpoints import BranchCheckpoints as SelectionCheckpoints


def build_parser():
    parser=common.build_parser()
    parser.description=__doc__
    parser.set_defaults(dataset="restaurant",readout="cls",baseline_regularization_weight=0.,log_step=20)
    parser.add_argument("--relation-reweight",action="store_true")
    parser.add_argument("--gate-relation-dim",type=int,default=32)
    parser.add_argument("--gate-hidden-dim",type=int,default=32)
    parser.add_argument("--gate-strength",type=float,default=.5)
    parser.add_argument("--save-last",action="store_true")
    parser.add_argument("--comparison-profile",choices=("v1","v1_relation_gate"))
    return parser


def dataset_options(config,tokenizer,vocab,dependency_vocab):
    options=common.dataset_options(config,tokenizer,vocab)
    options.update(dependency_vocab=dependency_vocab or None,typed_enabled=config["relation_reweight"])
    return options


def _counterfactual(model,path,tokenizer,options,device,dataset):
    return evaluate_counterfactual(model,path,tokenizer,options,device,dataset_name=dataset)


def run(args):
    if args.evaluation_protocol!="baseline" or args.val_file is not None:
        raise ValueError("Keep full training set and periodic test selection")
    if args.readout!="cls" or args.baseline_regularization_weight!=0 or args.no_routing:
        raise ValueError("Keep original v1 CLS/routing and old-loss weight zero")
    profile="v1_relation_gate" if args.relation_reweight else "v1"
    if args.comparison_profile not in (None,profile):raise ValueError("Profile differs from gate activation")
    args.comparison_profile=profile
    args.eval_interval_unit="optimizer_step"
    args.additive_relation_messages=False
    args.base_model="frozen_v1"
    args.dependency_feature_usage="edge_gate_only" if args.relation_reweight else "none"
    if args.output is None:
        args.output=PROJECT_ROOT/"runs_v1_relation_gate"/(("gate_" if args.relation_reweight else "v1_")+uuid.uuid4().hex[:10])
    return _run(args,extra_model_kwargs={k:getattr(args,k) for k in
                ("relation_reweight","gate_relation_dim","gate_hidden_dim","gate_strength")})


def main(argv=None):return run(build_parser().parse_args(argv))


def validate_checkpoint(checkpoint):
    config,kwargs=checkpoint["config"],checkpoint["model_kwargs"]
    for key in ("readout","baseline_regularization_weight","relation_reweight","num_dependency_relations",
                "gate_relation_dim","gate_hidden_dim","gate_strength"):
        if key not in config or config[key]!=kwargs.get(key):raise ValueError("Invalid checkpoint metadata: "+key)
    if config.get("additive_relation_messages") is not False or kwargs["readout"]!="cls" or kwargs["baseline_regularization_weight"]!=0:
        raise ValueError("Checkpoint must be pure v1 with optional edge gating only")
    if kwargs["relation_reweight"]:
        validate_dependency_vocab(checkpoint["dependency_vocab"])
        if len(checkpoint["dependency_vocab"])!=kwargs["num_dependency_relations"]:raise ValueError("Vocabulary size mismatch")
    elif checkpoint["dependency_vocab"]!={} or kwargs["num_dependency_relations"]!=0:
        raise ValueError("Ungated v1 must not use parser embeddings")


def _run(args, *, objective_factory=None, variant=VARIANT,
        model_factory=V1RelationGateDAGF, extra_model_kwargs=None):
    args = resolve_defaults(args)
    for name in ("epochs", "batch_size", "eval_batch_size", "grad_accum_steps", "max_length", "max_amr_edges", "log_step"):
        if getattr(args, name) < 1:
            raise ValueError(name + " must be positive")
    if args.num_workers < 0 or args.max_steps < 0 or args.latency_sentences < 0:
        raise ValueError("Worker, debug-step and latency counts must be nonnegative")
    if args.latency_repeats < 1:
        raise ValueError("latency_repeats must be positive")
    if min(args.lr, args.added_lr) <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rates must be positive and weight decay nonnegative")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = _check_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    folder = PROJECT_ROOT / "dataset" / DATASET_FOLDERS[args.dataset]
    args.train_file = args.train_file or folder / "train_with_amr.json"
    args.test_file = args.test_file or folder / "test_with_amr.json"
    if args.evaluation_protocol == "baseline":
        if args.val_file is not None:
            raise ValueError("The baseline protocol uses full train/test without --val-file")
        fields = ("train_file", "test_file")
        selection_split, selection_file = "test", args.test_file
    else:
        if args.val_file is None:
            candidate = folder / "valid_with_amr.json"
            if args.dataset != "mams" or not candidate.is_file():
                raise ValueError("The validation protocol requires an independent --val-file")
            args.val_file = candidate
        fields = ("train_file", "val_file", "test_file")
        selection_split, selection_file = "validation", args.val_file
    paths = [Path(getattr(args, name)).resolve() for name in fields]
    if len(set(paths)) != len(fields):
        raise ValueError("Train and evaluation split files must be distinct")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (args.roberta_path / "config.json").is_file() and (args.roberta_path / "roberta" / "config.json").is_file():
        args.roberta_path = args.roberta_path / "roberta"
    tokenizer = RobertaTokenizerFast.from_pretrained(args.roberta_path, add_prefix_space=True, local_files_only=True)
    vocab = build_amr_vocab(args.train_file)
    dependency_vocab = build_dependency_vocab(args.train_file) if args.relation_reweight else {}
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["model_variant"] = variant
    config["num_dependency_relations"] = len(dependency_vocab)
    config["reference_model"] = "frozen_v1_snapshot_20261029"
    config["data_adapter"] = "twitter_v1_tokenizer_coverage" if args.dataset == "twitter" else "frozen_v1"
    options = dataset_options(config, tokenizer, vocab, dependency_vocab)
    trainset = _dataset_class(args.dataset)(args.train_file, **options)
    selection_set = _dataset_class(args.dataset)(selection_file, **options)
    model_kwargs = dict(
        roberta_path=str(args.roberta_path.resolve()), hidden_dim=args.hidden_dim, readout=args.readout,
        lstm_hidden=args.lstm_hidden, amr_dim=args.amr_dim, num_layers=args.num_layers,
        attention_heads=args.attention_heads, dropout=args.dropout, num_amr_relations=len(vocab),
        alpha=args.alpha, beta=args.beta, kl_weight=args.kl_weight,
        gradient_checkpointing=args.gradient_checkpointing, use_amr=not args.no_amr,
        use_agf=not args.no_agf, use_routing=not args.no_routing,
        use_inter_aspect=not args.no_inter_aspect, use_structure_prior=not args.no_structure_prior,
        sparsity_weight=args.sparsity_weight, exclusivity_weight=args.exclusivity_weight,
        sharing_weight=args.sharing_weight, baseline_regularization_weight=args.baseline_regularization_weight,
    )
    extra_model_kwargs = dict(extra_model_kwargs or {})
    extra_model_kwargs["num_dependency_relations"] = len(dependency_vocab)
    if set(extra_model_kwargs) & set(model_kwargs):
        raise ValueError("Extension model kwargs must not override the reference model options")
    model_kwargs.update(extra_model_kwargs)
    model = model_factory(**model_kwargs).to(device)
    if args.max_length > model.encoder.config.max_position_embeddings - model.pad_token_id - 1:
        raise ValueError("max_length exceeds RoBERTa position capacity")
    output = args.output or PROJECT_ROOT / "runs_typed_syntax" / (args.dataset + "_" + uuid.uuid4().hex[:10])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config["output"] = str(output)
    write_json(output / "config.json", config)
    write_json(output / "dependency_vocab.json", dependency_vocab)
    write_json(output / "data_summary.json", {"train": trainset.stats, selection_split: selection_set.stats,
                                             "selection_split": selection_split,
                                             "evaluation_protocol": args.evaluation_protocol})
    objective = objective_factory(args, tokenizer) if objective_factory is not None else None
    extra_components = tuple(getattr(objective, "component_names", ()))
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = args.lr if name.startswith(("encoder.", "mlm.")) else args.added_lr
        decay = args.weight_decay if parameter.ndim > 1 else 0.0
        groups.setdefault((lr, decay), []).append(parameter)
    optimizer = torch.optim.AdamW([{"params": params, "lr": lr, "weight_decay": decay}
                                   for (lr, decay), params in groups.items()])
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, generator=generator,
                        num_workers=args.num_workers, collate_fn=sentence_collate)
    steps, stopped = 0, False
    base_payload = dict(variant=variant, config=config, model_kwargs=model_kwargs, amr_vocab=vocab, dependency_vocab=dependency_vocab)
    checkpoints = SelectionCheckpoints(output, base_payload, args.evaluation_protocol, selection_split)
    print("Sentence-level routing run (%s selection): %s" % (selection_split, output), flush=True)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        components = {name: 0.0 for name in ("sentence_ce", "sparsity_loss", "coverage_loss",
                                             "route_entropy_loss", "exclusivity_loss", "sharing_loss")}
        if set(extra_components) & set(components):
            raise ValueError("Objective component names collide with base metrics")
        components.update({name: 0.0 for name in extra_components})
        last_evaluated_step = None

        def record_selection(batch_index):
            report, predictions = evaluate(model, selection_set, device, args.eval_batch_size, args.num_workers)
            report.update(epoch=epoch + 1, batch=batch_index + 1, optimizer_steps=steps,
                          train_loss=loss_sum / (batch_index + 1), selection_split=selection_split,
                          evaluation_protocol=args.evaluation_protocol, debug_limited=bool(args.max_steps),
                          eval_interval_unit="optimizer_step", log_step=args.log_step)
            report["train_components"] = {name: value / (batch_index + 1) for name, value in components.items()}
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
            checkpoints.record(model, report, predictions)
            print(json.dumps({key: report[key] for key in
                              ("epoch", "optimizer_steps", "selection_split", "train_loss", "accuracy", "macro_f1")}), flush=True)
            return report

        for index, batch in enumerate(loader):
            labels = batch["labels"].to(device)
            inputs = model_inputs(batch, device)
            if objective is None:
                prediction = model(inputs, labels=labels)
                ce = sentence_cross_entropy(prediction["logits"], labels, inputs["aspect_valid"])
                loss = ce + prediction["regularization_loss"]
                extra_values = {}
            else:
                # Optional auxiliary gold stays in objective kwargs, separate
                # from the model's text/graph inputs. Existing objectives use
                # the unchanged three-argument path.
                prepare_batch = getattr(objective, "prepare_batch", None)
                if prepare_batch is None:
                    prediction, ce, loss, extra_values = objective(model, inputs, labels)
                else:
                    supervised = prepare_batch(batch, device)
                    if not isinstance(supervised, dict):
                        raise TypeError("Objective prepare_batch must return supervised keyword arguments")
                    prediction, ce, loss, extra_values = objective(model, inputs, labels, **supervised)
                if set(extra_values) != set(extra_components):
                    raise ValueError("Objective metrics differ from declared component_names")
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite routing training loss")
            group_start = (index // args.grad_accum_steps) * args.grad_accum_steps
            group_size = min(args.grad_accum_steps, len(loader) - group_start)
            (loss / group_size).backward()
            loss_sum += float(loss.detach())
            components["sentence_ce"] += float(ce.detach())
            for name in components:
                if name != "sentence_ce":
                    value = extra_values[name] if name in extra_values else prediction[name]
                    components[name] += float(value.detach())
            boundary = (index + 1) % args.grad_accum_steps == 0 or index + 1 == len(loader)
            if not boundary:
                continue
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            if steps % args.log_step == 0:
                record_selection(index)
                last_evaluated_step = steps
            if args.max_steps and steps >= args.max_steps:
                stopped = True
                break
        if last_evaluated_step != steps:
            record_selection(index)
        if args.save_last:
            save_checkpoint(output / "last.pt", dict(base_payload,
                            model_state={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                            epoch=epoch + 1, optimizer_steps=steps,best_scores=checkpoints.score_summary()))
        if stopped:
            break
    selected = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model_state"], strict=True)
    testset = selection_set if selection_split == "test" else _dataset_class(args.dataset)(args.test_file, **options)
    report, predictions = evaluate(model, testset, device, args.eval_batch_size,
                                    args.num_workers, include_routes=args.export_routes,
                                    evidence_annotations=args.evidence_annotations)
    report["selected_checkpoint"] = "best_acc.pt" if selection_split == "test" else "best_f1.pt"
    report["model_variant"] = variant
    report["readout"] = args.readout
    report["evaluation_protocol"] = args.evaluation_protocol
    report["selection_split"] = selection_split
    report["best_scores"] = checkpoints.score_summary()
    report["data_stats"] = testset.stats
    report["debug_limited"] = bool(args.max_steps)
    report["counterfactual"] = _counterfactual(model, args.counterfactual_file,
                                                       tokenizer, options, device, args.dataset)
    report["latency"] = benchmark_latency(model, testset, device,
                                          repeats=args.latency_repeats,
                                          max_sentences=args.latency_sentences) if args.latency_sentences else {"status": "not_evaluated"}
    write_json(output / "test_report.json", report)
    write_json(output / "data_summary.json", {"train": trainset.stats, selection_split: selection_set.stats,
                                             "test": testset.stats, "selection_split": selection_split,
                                             "evaluation_protocol": args.evaluation_protocol})
    write_json(output / "test_predictions.json", predictions)
    write_json(output / "completion.json", dict(status="complete" if not args.max_steps else "debug_complete",
               comparison_profile="v1_relation_gate" if args.relation_reweight else "v1",
               seed=args.seed,epochs_completed=epoch+1,optimizer_steps=steps,
               eval_interval_unit="optimizer_step",log_step=args.log_step,debug_limited=bool(args.max_steps)))
    print("Finished; reports and checkpoints: " + str(output), flush=True)
    return output

def evaluate_main(argv=None, *, variant=VARIANT, model_factory=V1RelationGateDAGF,
                  checkpoint_validator=validate_checkpoint):
    parser = argparse.ArgumentParser(description="Evaluate or predict with a routing checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New report directory")
    parser.add_argument("--roberta-path", type=Path, help="Relocated identical backbone/tokenizer directory")
    parser.add_argument("--lexicon-dir", type=Path, help="Relocated identical opinion word lists")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--counterfactual-file", type=Path)
    parser.add_argument("--evidence-annotations", type=Path)
    parser.add_argument("--latency-sentences", type=int, default=100)
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument("--export-routes", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args(argv)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = _check_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("variant") != variant:
        raise ValueError("Checkpoint variant mismatch; use the matching extension evaluator")
    if checkpoint_validator is not None:
        checkpoint_validator(checkpoint)
    config, kwargs = dict(checkpoint["config"]), dict(checkpoint["model_kwargs"])
    if args.roberta_path:
        config["roberta_path"] = kwargs["roberta_path"] = str(args.roberta_path.resolve())
    if args.lexicon_dir:
        config["lexicon_dir"] = str(args.lexicon_dir.resolve())
    tokenizer = RobertaTokenizerFast.from_pretrained(config["roberta_path"], add_prefix_space=True, local_files_only=True)
    options = dataset_options(config, tokenizer, checkpoint["amr_vocab"], checkpoint["dependency_vocab"])
    dataset = _dataset_class(config["dataset"])(args.data_file, require_labels=False, **options)
    if "readout" not in kwargs or kwargs["readout"] != config.get("readout"):
        raise ValueError("Missing or inconsistent checkpoint readout metadata")
    model = model_factory(**kwargs).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    report, predictions = evaluate(model, dataset, device, args.batch_size, include_routes=args.export_routes,
                                    evidence_annotations=args.evidence_annotations)
    report["data_stats"] = dataset.stats
    report["model_variant"] = variant
    report["readout"] = kwargs["readout"]
    report["counterfactual"] = _counterfactual(model, args.counterfactual_file, tokenizer, options, device, config["dataset"])
    report["latency"] = benchmark_latency(model, dataset, device, repeats=args.latency_repeats,
                                          max_sentences=args.latency_sentences) if args.latency_sentences else {"status": "not_evaluated"}
    write_json(args.output / "report.json", report)
    write_json(args.output / "predictions.json", predictions)
    print("Saved evaluation: " + str(args.output.resolve()), flush=True)
    return args.output
