"""Additive training entry with the existing baseline's train/test protocol."""

import argparse
import json
from pathlib import Path
import random
import uuid

import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast

from . import VARIANT
from ..v1_reference.data import DATASET_FOLDERS, SentenceABSADataset, build_amr_vocab, sentence_collate
from ..twitter_v1.data import TwitterABSADataset
from ..v1_reference.evaluation import benchmark_latency, evaluate, model_inputs
from ..v1_reference import evaluation as v1_evaluation
from ..twitter_v1 import evaluation as twitter_evaluation
from .model import TargetReadoutDAGF, sentence_cross_entropy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DEFAULTS = {
    "restaurant": {"epochs": 50, "alpha": 0.8, "beta": 0.6, "kl_weight": 1.5},
    "laptop": {"epochs": 30, "alpha": 0.5, "beta": 0.4, "kl_weight": 1.6},
    "twitter": {"epochs": 30, "alpha": 0.6, "beta": 0.6, "kl_weight": 1.3},
    # MAMS has no entry in the original run.sh; keep an explicit 30-epoch setting.
    "mams": {"epochs": 30, "alpha": 0.8, "beta": 0.6, "kl_weight": 1.5},
}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, payload):
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_FOLDERS), default="mams")
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--val-file", type=Path)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--evaluation-protocol", choices=("baseline", "validation"), default="baseline",
                        help="baseline: full train, periodic test, best test Acc/F1; validation: separate selection split")
    parser.add_argument("--log-step", type=int, default=100,
                        help="Evaluate every N optimizer updates, and at the end of each epoch, as the current baseline")
    parser.add_argument("--roberta-path", type=Path, default=PROJECT_ROOT / "roberta" / "roberta")
    parser.add_argument("--lexicon-dir", type=Path, default=PROJECT_ROOT / "opinion-lexicon-English")
    parser.add_argument("--no-lexicon", action="store_true")
    parser.add_argument("--output", type=Path, help="A new run directory; never overwrites an existing run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--epochs", type=int, help="Default follows the current run.sh: restaurant 50, laptop 30")
    parser.add_argument("--batch-size", type=int, default=16, help="Sentences per batch; the original baseline batches aspect instances")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, help="Whole-sentence upper bound: laptop 512, other supplied datasets 384")
    parser.add_argument("--max-amr-edges", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--lstm-hidden", type=int, default=300)
    parser.add_argument("--amr-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="Default: Twitter 1e-5; other datasets 2e-5")
    parser.add_argument("--added-lr", type=float, help="Default equals --lr, matching the baseline's unified learning rate")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--kl-weight", type=float)
    parser.add_argument("--weight-decay", type=float, help="Default: Twitter 1e-4; other datasets 1e-2")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--sparsity-weight", type=float, default=0.01)
    parser.add_argument("--exclusivity-weight", type=float, default=0.01)
    parser.add_argument("--sharing-weight", type=float, default=0.01)
    parser.add_argument("--baseline-regularization-weight", type=float, default=0.0,
                        help="Optional old KL+aux terms; default objective is sentence CE plus routing penalties")
    parser.add_argument("--readout", choices=("aspect", "cls"), default="aspect",
                        help="Parameter-matched classifier input: fused+aspect query or frozen fused+CLS")
    parser.add_argument("--no-routing", action="store_true")
    parser.add_argument("--no-inter-aspect", action="store_true")
    parser.add_argument("--no-structure-prior", action="store_true")
    parser.add_argument("--no-amr", action="store_true")
    parser.add_argument("--no-agf", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--counterfactual-file", type=Path)
    parser.add_argument("--evidence-annotations", type=Path,
                        help="Optional independently verified opinion spans for test-only gold buckets")
    parser.add_argument("--latency-sentences", type=int, default=100)
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument("--export-routes", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="Smoke/debug limit; 0 means full training")
    parser.add_argument("--threads", type=int, default=0, help="Optional CPU thread count")
    return parser


def dataset_options(config, tokenizer, vocab):
    return dict(tokenizer=tokenizer, max_length=config["max_length"], amr_vocab=vocab,
                lexicon_dir=None if config["no_lexicon"] else config["lexicon_dir"],
                max_amr_edges=config["max_amr_edges"])


def _check_device(device):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the active Python environment")
    return device


def resolve_defaults(args):
    for name, value in DATASET_DEFAULTS[args.dataset].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.max_length is None:
        args.max_length = 512 if args.dataset == "laptop" else 384
    if args.lr is None:
        args.lr = 1e-5 if args.dataset == "twitter" else 2e-5
    if args.weight_decay is None:
        args.weight_decay = 1e-4 if args.dataset == "twitter" else 1e-2
    if args.added_lr is None:
        args.added_lr = args.lr
    return args


class SelectionCheckpoints:
    """Keep the two baseline maxima and their actual paired checkpoint metrics."""

    def __init__(self, output, base_payload, protocol, selection_split):
        self.output, self.base_payload = output, base_payload
        self.protocol, self.selection_split = protocol, selection_split
        self.best_acc = self.best_f1 = None

    def payload(self, model, summary):
        return dict(self.base_payload,
                    model_state={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                    selection=summary, epoch=summary["epoch"], optimizer_steps=summary["optimizer_steps"])

    def record(self, model, summary, predictions):
        acc_key = (summary["accuracy"], summary["macro_f1"])
        f1_key = (summary["macro_f1"], summary["accuracy"])
        improve_acc = self.best_acc is None or acc_key > (self.best_acc["accuracy"], self.best_acc["macro_f1"])
        improve_f1 = self.best_f1 is None or f1_key > (self.best_f1["macro_f1"], self.best_f1["accuracy"])
        payload = self.payload(model, summary) if improve_acc or improve_f1 else None
        primary = "best_acc" if self.protocol == "baseline" else "best_f1"
        for name, improved in (("best_acc", improve_acc), ("best_f1", improve_f1)):
            if not improved:
                continue
            setattr(self, name, dict(summary))
            save_checkpoint(self.output / (name + ".pt"), payload)
            write_json(self.output / (name + ".json"), summary)
            write_json(self.output / (name + "_predictions.json"), predictions)
            if name == primary:
                # Compatibility alias used by evaluate_routing.py commands.
                save_checkpoint(self.output / "best.pt", payload)
        write_json(self.output / "best_scores.json", self.score_summary())

    def score_summary(self):
        result = {"evaluation_protocol": self.protocol, "selection_split": self.selection_split,
                  "best_accuracy_checkpoint": "best_acc.pt", "best_macro_f1_checkpoint": "best_f1.pt",
                  "best_accuracy": self.best_acc, "best_macro_f1": self.best_f1,
                  "maxima_may_come_from_different_checkpoints": True}
        if self.protocol == "baseline" and self.best_acc is not None:
            result.update(max_test_acc=self.best_acc["accuracy"], max_f1=self.best_f1["macro_f1"],
                          f1_at_max_test_acc=self.best_acc["macro_f1"])
        return result


def _dataset_class(dataset):
    return TwitterABSADataset if dataset == "twitter" else SentenceABSADataset


def _counterfactual(model, path, tokenizer, options, device, dataset):
    evaluator = twitter_evaluation if dataset == "twitter" else v1_evaluation
    return evaluator.evaluate_counterfactual(model, path, tokenizer, options, device)


def run(args, *, objective_factory=None, variant=VARIANT,
        model_factory=TargetReadoutDAGF, extra_model_kwargs=None):
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
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["model_variant"] = variant
    config["reference_model"] = "frozen_v1_snapshot_20261029"
    config["data_adapter"] = "twitter_v1_tokenizer_coverage" if args.dataset == "twitter" else "frozen_v1"
    options = dataset_options(config, tokenizer, vocab)
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
    if set(extra_model_kwargs) & set(model_kwargs):
        raise ValueError("Extension model kwargs must not override the reference model options")
    model_kwargs.update(extra_model_kwargs)
    model = model_factory(**model_kwargs).to(device)
    if args.max_length > model.encoder.config.max_position_embeddings - model.pad_token_id - 1:
        raise ValueError("max_length exceeds RoBERTa position capacity")
    output = args.output or PROJECT_ROOT / "runs_target_readout" / (args.dataset + "_" + uuid.uuid4().hex[:10])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config["output"] = str(output)
    write_json(output / "config.json", config)
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
    base_payload = dict(variant=variant, config=config, model_kwargs=model_kwargs, amr_vocab=vocab)
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
                          evaluation_protocol=args.evaluation_protocol, debug_limited=bool(args.max_steps))
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
        save_checkpoint(output / "last.pt", dict(base_payload,
                        model_state={name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                        epoch=epoch + 1, optimizer_steps=steps, optimizer_state=optimizer.state_dict(),
                        best_scores=checkpoints.score_summary()))
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
    print("Finished; reports and checkpoints: " + str(output), flush=True)
    return output


def main(argv=None):
    return run(build_parser().parse_args(argv))


def evaluate_main(argv=None, *, variant=VARIANT, model_factory=TargetReadoutDAGF,
                  checkpoint_validator=None):
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
    options = dataset_options(config, tokenizer, checkpoint["amr_vocab"])
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
