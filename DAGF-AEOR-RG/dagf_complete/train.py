"""Default DAGF training, paired checkpoints, target stopping and interruption.

Legacy training sources are retained in legacy_backups/. The state_dict
directory and graph caches are not overwritten by this training backend.
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
import random
import signal
import sys
from datetime import datetime
import uuid

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler
from sklearn import metrics as sklearn_metrics
from transformers import RobertaTokenizerFast

from . import VARIANT
from .data import CompleteABSADataset, DATASET_FOLDERS, build_amr_vocab
from .model import CompleteDAGF


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "dataset": "restaurant", "seed": 1000, "cuda": "0", "device": "cuda",
    "num_epoch": 50, "batch_size": 16, "eval_batch_size": 16,
    "bert_lr": 2e-5, "weight_decay": 1e-2, "adam_epsilon": 1e-8,
    "dropout": 0.3, "hidden_dim": 768, "lstm_hidden": 300,
    "num_layers": 1, "attention_heads": 8, "max_length": 100,
    "alpha": 0.8, "beta": 0.6, "kl_weight": 1.5,
    "aux_loss_weight": 0.1, "amr_dim": 128, "max_amr_edges": 2048,
    "log_step": 100, "grad_accum_steps": 1, "max_grad_norm": 1.0,
    "save_every_steps": 0, "num_workers": 0,
    "roberta_path": "./roberta", "lexicon_dir": "./opinion-lexicon-English",
    "output_root": "./runs_complete", "train_file": None, "test_file": None,
    "val_file": None, "no_amr": False, "no_agf": False,
    "gradient_checkpointing": False, "gradient_detach": True,
    "stop_at_acc": None, "stop_at_f1": None,
}
RESUME_OVERRIDES = {
    "num_epoch", "cuda", "device", "eval_batch_size", "log_step",
    "save_every_steps", "num_workers", "stop_at_acc", "stop_at_f1",
}
LOGGER = logging.getLogger("dagf_complete")


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", help="Full last.pt checkpoint from this new entry")
    parser.add_argument("--dataset", choices=sorted(DATASET_FOLDERS))
    parser.add_argument("--model_name", choices=("newbert", "complete", "dagf_complete"),
                        help="Compatibility argument; all names use the completed DAGF model")
    parser.add_argument("--lexicon", choices=sorted(DATASET_FOLDERS),
                        help="Compatibility argument; when supplied it must match dataset")
    parser.add_argument("--gcn_dropout", type=float,
                        help="Compatibility argument; must match the unified dropout value")
    for name in ("seed", "num_epoch", "batch_size", "eval_batch_size", "hidden_dim",
                 "num_layers", "attention_heads", "max_length", "amr_dim",
                 "max_amr_edges", "log_step", "grad_accum_steps", "save_every_steps", "num_workers"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--lstm_hidden", "--rnn_hidden", dest="lstm_hidden", type=int)
    parser.add_argument("--bert_lr", "--learning_rate", dest="bert_lr", type=float)
    parser.add_argument("--weight_decay", "--l2reg", dest="weight_decay", type=float)
    parser.add_argument("--kl_weight", "--gama", dest="kl_weight", type=float)
    parser.add_argument("--dropout", "--bert_dropout", dest="dropout", type=float)
    for name in ("adam_epsilon", "alpha", "beta", "aux_loss_weight",
                 "max_grad_norm", "stop_at_acc", "stop_at_f1"):
        parser.add_argument("--" + name, type=float)
    for name in ("cuda", "device", "roberta_path", "lexicon_dir", "output_root",
                 "train_file", "test_file", "val_file"):
        parser.add_argument("--" + name)
    for name in ("no_amr", "no_agf", "gradient_checkpointing"):
        parser.add_argument("--" + name, action="store_true", default=None)
    parser.add_argument("--differentiate_auxiliary_gradient", dest="gradient_detach",
                        action="store_false", default=None)
    supplied = vars(parser.parse_args())
    resume_path = supplied.pop("resume")
    compatibility = {name: supplied.pop(name) for name in ("model_name", "lexicon", "gcn_dropout")}
    checkpoint = None
    config = dict(DEFAULTS)
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("variant") != VARIANT or checkpoint.get("kind") != "training_state":
            raise ValueError("--resume requires this variant's full last.pt, not a legacy or best-weights file")
        config.update(checkpoint["config"])
        for name, value in supplied.items():
            if value is not None and name not in RESUME_OVERRIDES and value != config.get(name):
                raise ValueError("Cannot change %s while resuming; use the saved configuration" % name)
        for name in RESUME_OVERRIDES:
            if supplied.get(name) is not None:
                config[name] = supplied[name]
    else:
        config.update({name: value for name, value in supplied.items() if value is not None})
    if compatibility["lexicon"] is not None and compatibility["lexicon"] != config["dataset"]:
        parser.error("--lexicon must match --dataset; use --lexicon_dir for a word-list directory")
    if compatibility["gcn_dropout"] is not None and compatibility["gcn_dropout"] != config["dropout"]:
        parser.error("The new model uses unified dropout; remove --gcn_dropout or match --dropout")
    for name in ("stop_at_acc", "stop_at_f1"):
        value = config[name]
        if value is not None:
            value = value / 100.0 if value > 1.0 else value
            if not 0 <= value <= 1:
                raise ValueError(name + " must be a fraction in [0,1] or a percentage in [0,100]")
            config[name] = value
    for name in ("num_epoch", "batch_size", "eval_batch_size", "log_step", "grad_accum_steps"):
        if config[name] < 1:
            raise ValueError(name + " must be positive")
    if config["save_every_steps"] < 0 or config["num_workers"] < 0:
        raise ValueError("save_every_steps and num_workers cannot be negative")
    folder = DATASET_FOLDERS[config["dataset"]]
    for field, split in (("train_file", "train"), ("test_file", "test")):
        if config[field] is None:
            config[field] = str(PROJECT_ROOT / "dataset" / folder / (split + "_with_amr.json"))
    for field in ("train_file", "test_file", "val_file", "roberta_path", "lexicon_dir", "output_root"):
        if config[field] is not None:
            config[field] = str(Path(config[field]).resolve())
    return argparse.Namespace(**config), checkpoint, resume_path


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; exact RNG resume is unavailable")
        torch.cuda.set_rng_state_all(state["cuda"])


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def _atomic_torch(path, value):
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    try:
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    except BaseException:
        # A previously completed checkpoint stays intact if serialization fails.
        if temporary.exists():
            temporary.unlink()
        raise


class EpochSampler(Sampler):
    """A deterministic permutation with an offset for safe mid-epoch resume."""

    def __init__(self, size, seed, epoch, start_sample=0):
        generator = torch.Generator().manual_seed(seed + epoch)
        self.indices = torch.randperm(size, generator=generator).tolist()[start_sample:]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class StopRequest:
    def __init__(self):
        self.requested = False
        self.previous = {}

    def _handler(self, signum, frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        LOGGER.warning("Stop requested. Finishing the current optimizer update, then saving last.pt.")

    def install(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous[signum] = signal.signal(signum, self._handler)

    def restore(self):
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)


def _loader(dataset, args, sampler=None, evaluation=False):
    # Supplying a dedicated generator prevents loader creation from changing
    # the model's global RNG stream during an interrupted/resumed epoch.
    return DataLoader(
        dataset, batch_size=args.eval_batch_size if evaluation else args.batch_size,
        sampler=sampler, shuffle=False, num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        generator=torch.Generator().manual_seed(args.seed),
    )


def _model_batch(batch, device):
    labels = batch["polarity"].to(device)
    inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items() if key != "polarity"
    }
    return inputs, labels


def _evaluate(model, loader, device, stop=None):
    model.eval()
    targets, predictions, rows = [], [], []
    with torch.no_grad():
        for batch in loader:
            if stop is not None and stop.requested:
                return None
            inputs, labels = _model_batch(batch, device)
            # Gold labels are held outside the predictive forward pass.
            probabilities = model(inputs)["logits"].softmax(dim=-1)
            guessed = probabilities.argmax(dim=-1)
            target_list = labels.cpu().tolist()
            prediction_list = guessed.cpu().tolist()
            probabilities_list = probabilities.cpu().tolist()
            indices = batch["sample_index"].tolist()
            targets.extend(target_list)
            predictions.extend(prediction_list)
            for index, target, prediction, probability in zip(indices, target_list, prediction_list, probabilities_list):
                rows.append({"sample_index": index, "target": target,
                             "prediction": prediction, "probabilities": probability})
    summary = {
        "accuracy": float(sklearn_metrics.accuracy_score(targets, predictions)),
        "macro_f1": float(sklearn_metrics.f1_score(targets, predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "examples": len(targets),
        "classification_report": sklearn_metrics.classification_report(
            targets, predictions, labels=[0, 1, 2],
            target_names=["positive", "negative", "neutral"], zero_division=0, output_dict=True),
        "confusion_matrix": sklearn_metrics.confusion_matrix(targets, predictions, labels=[0, 1, 2]).tolist(),
    }
    return summary, rows


class Checkpoints:
    def __init__(self, run_dir, args, model, optimizer, vocabulary, model_kwargs, restored=None):
        self.run_dir, self.args = run_dir, args
        self.model, self.optimizer = model, optimizer
        self.vocabulary, self.model_kwargs = vocabulary, model_kwargs
        self.best_acc = restored.get("best_acc") if restored else None
        self.best_f1 = restored.get("best_f1") if restored else None

    def weights(self, name, summary, predictions):
        _atomic_torch(self.run_dir / (name + ".pt"), {
            "variant": VARIANT, "kind": "model_weights",
            "config": vars(self.args), "model_kwargs": self.model_kwargs,
            "amr_vocab": self.vocabulary, "metrics": summary,
            "model_state": _cpu_tree(self.model.state_dict()),
        })
        _atomic_json(self.run_dir / (name + ".json"), summary)
        _atomic_json(self.run_dir / (name + "_predictions.json"), predictions)
        LOGGER.info("saved %s.pt: accuracy=%.4f macro_f1=%.4f", name, summary["accuracy"], summary["macro_f1"])

    def record(self, summary, predictions):
        with (self.run_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(summary, ensure_ascii=False, allow_nan=False) + "\n")
        acc_key = (summary["accuracy"], summary["macro_f1"])
        f1_key = (summary["macro_f1"], summary["accuracy"])
        if self.best_acc is None or acc_key > (self.best_acc["accuracy"], self.best_acc["macro_f1"]):
            self.weights("best_acc", summary, predictions)
            self.best_acc = dict(summary)
        if self.best_f1 is None or f1_key > (self.best_f1["macro_f1"], self.best_f1["accuracy"]):
            self.weights("best_f1", summary, predictions)
            self.best_f1 = dict(summary)

    def last(self, next_epoch, next_batch, global_step, reason):
        _atomic_torch(self.run_dir / "last.pt", {
            "variant": VARIANT, "kind": "training_state", "reason": reason,
            "config": vars(self.args), "model_kwargs": self.model_kwargs,
            "amr_vocab": self.vocabulary,
            "model_state": _cpu_tree(self.model.state_dict()),
            "optimizer_state": _cpu_tree(self.optimizer.state_dict()),
            "next_epoch": next_epoch, "next_batch": next_batch,
            "global_step": global_step, "rng_state": _rng_state(),
            "best_acc": self.best_acc, "best_f1": self.best_f1,
        })
        _atomic_json(self.run_dir / "status.json", {
            "reason": reason, "next_epoch": next_epoch + 1,
            "next_batch_zero_based": next_batch, "global_step": global_step,
            "best_acc": self.best_acc, "best_f1": self.best_f1,
        })
        LOGGER.info("saved last.pt: reason=%s next_epoch=%d next_batch=%d", reason, next_epoch + 1, next_batch)


def _target_reached(summary, args):
    provided = args.stop_at_acc is not None or args.stop_at_f1 is not None
    # Targets quoted from papers/logs are percentages rounded to two decimals.
    # The unrounded metric is retained in every checkpoint and JSON record.
    def reaches(actual, target):
        return target is None or round(actual * 100.0, 2) >= round(target * 100.0, 2)
    return provided and (
        reaches(summary["accuracy"], args.stop_at_acc)
    ) and (
        reaches(summary["macro_f1"], args.stop_at_f1)
    )


def _train(model, optimizer, trainset, evaluation_loader, args, device, checkpoints, restored, evaluation_split):
    start_epoch = restored["next_epoch"] if restored else 0
    start_batch = restored["next_batch"] if restored else 0
    global_step = restored["global_step"] if restored else 0
    total_batches = math.ceil(len(trainset) / args.batch_size)
    stop = StopRequest()
    stop.install()
    criterion = nn.CrossEntropyLoss()
    try:
        for epoch in range(start_epoch, args.num_epoch):
            offset = start_batch if epoch == start_epoch else 0
            sampler = EpochSampler(len(trainset), args.seed, epoch, offset * args.batch_size)
            loader = _loader(trainset, args, sampler=sampler)
            optimizer.zero_grad(set_to_none=True)
            model.train()
            last_evaluated_step = None
            LOGGER.info("epoch: %d/%d; starting batch: %d/%d", epoch + 1, args.num_epoch, offset + 1, total_batches)
            if stop.requested:
                checkpoints.last(epoch, offset, global_step, "interrupted")
                return
            for batch_index, batch in enumerate(loader, start=offset):
                model.train()
                inputs, labels = _model_batch(batch, device)
                output = model(inputs, labels=labels)
                loss = criterion(output["logits"], labels) + output["regularization_loss"]
                if not torch.isfinite(loss).item():
                    raise RuntimeError("Non-finite loss; existing completed checkpoints remain available")
                group_start = (batch_index // args.grad_accum_steps) * args.grad_accum_steps
                group_size = min(args.grad_accum_steps, total_batches - group_start)
                (loss / group_size).backward()
                boundary = (batch_index + 1) % args.grad_accum_steps == 0 or batch_index + 1 == total_batches
                if not boundary:
                    continue
                if args.max_grad_norm > 0:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if not torch.isfinite(norm).item():
                        raise RuntimeError("Non-finite gradients; existing completed checkpoints remain available")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                next_epoch, next_batch = epoch, batch_index + 1
                if next_batch >= total_batches:
                    next_epoch, next_batch = epoch + 1, 0
                if stop.requested:
                    checkpoints.last(next_epoch, next_batch, global_step, "interrupted")
                    return
                if global_step % args.log_step == 0:
                    result = _evaluate(model, evaluation_loader, device, stop)
                    if result is None:
                        checkpoints.last(next_epoch, next_batch, global_step, "interrupted")
                        return
                    summary, predictions = result
                    summary.update({"epoch": epoch + 1, "batch": batch_index + 1,
                                    "global_step": global_step, "dataset": args.dataset,
                                    "seed": args.seed, "selection_split": evaluation_split,
                                    "train_loss": float(loss.item()),
                                    "kl_loss": float(output["kl_loss"].item()),
                                    "auxiliary_loss": float(output["auxiliary_loss"].item())})
                    checkpoints.record(summary, predictions)
                    last_evaluated_step = global_step
                    LOGGER.info("loss: %.4f, %s_acc: %.4f, macro_f1: %.4f, kl: %.4f, auxiliary: %.4f",
                                loss.item(), evaluation_split, summary["accuracy"], summary["macro_f1"],
                                output["kl_loss"].item(), output["auxiliary_loss"].item())
                    if _target_reached(summary, args):
                        checkpoints.weights("baseline", summary, predictions)
                        checkpoints.last(next_epoch, next_batch, global_step, "target_reached")
                        LOGGER.info("Both requested score thresholds reached at the same evaluation; stopping.")
                        return
                if args.save_every_steps and global_step % args.save_every_steps == 0:
                    checkpoints.last(next_epoch, next_batch, global_step, "periodic")
            if last_evaluated_step != global_step:
                result = _evaluate(model, evaluation_loader, device, stop)
                if result is None:
                    checkpoints.last(epoch + 1, 0, global_step, "interrupted")
                    return
                summary, predictions = result
                summary.update({"epoch": epoch + 1, "batch": total_batches,
                                "global_step": global_step, "dataset": args.dataset,
                                "seed": args.seed, "selection_split": evaluation_split})
                checkpoints.record(summary, predictions)
                if _target_reached(summary, args):
                    checkpoints.weights("baseline", summary, predictions)
                    checkpoints.last(epoch + 1, 0, global_step, "target_reached")
                    return
            checkpoints.last(epoch + 1, 0, global_step, "epoch_completed")
            if stop.requested:
                return
        LOGGER.info("Training complete. best_acc=%s; best_f1=%s", checkpoints.best_acc, checkpoints.best_f1)
    finally:
        stop.restore()


def main():
    args, restored, resume_path = _arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; activate the existing dagf GPU environment")
    device = torch.device(args.device)
    _seed(args.seed)
    if resume_path:
        run_dir = Path(resume_path).resolve().parent
    else:
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        run_dir = Path(args.output_root) / args.dataset / ("seed_%d" % args.seed) / name
        run_dir.mkdir(parents=True, exist_ok=False)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "train.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)
    LOGGER.info("Default DAGF variant: %s; outputs: %s", VARIANT, run_dir)
    LOGGER.info("config: %s", json.dumps(vars(args), ensure_ascii=False))
    _atomic_json(run_dir / "config.json", vars(args))
    tokenizer = RobertaTokenizerFast.from_pretrained(args.roberta_path, add_prefix_space=True)
    vocabulary = restored["amr_vocab"] if restored else build_amr_vocab(args.train_file)
    _atomic_json(run_dir / "amr_vocab.json", vocabulary)
    dataset_kwargs = dict(tokenizer=tokenizer, max_length=args.max_length,
                          amr_vocab=vocabulary, lexicon_dir=args.lexicon_dir,
                          max_amr_edges=args.max_amr_edges)
    trainset = CompleteABSADataset(args.train_file, **dataset_kwargs)
    testset = CompleteABSADataset(args.test_file, **dataset_kwargs)
    if args.val_file:
        evaluation_set = CompleteABSADataset(args.val_file, **dataset_kwargs)
        evaluation_split = "validation"
    else:
        evaluation_set, evaluation_split = testset, "test"
    _atomic_json(run_dir / "data_summary.json", {
        "train": trainset.stats, "test": testset.stats,
        "selection": evaluation_set.stats, "selection_split": evaluation_split,
    })
    LOGGER.info("train data: %s", trainset.stats)
    LOGGER.info("evaluation data: %s", evaluation_set.stats)
    model_kwargs = dict(
        roberta_path=args.roberta_path, hidden_dim=args.hidden_dim,
        lstm_hidden=args.lstm_hidden, num_layers=args.num_layers,
        attention_heads=args.attention_heads, dropout=args.dropout,
        alpha=args.alpha, beta=args.beta, kl_weight=args.kl_weight,
        num_amr_relations=len(vocabulary), aux_loss_weight=args.aux_loss_weight,
        gradient_detach=args.gradient_detach,
        gradient_checkpointing=args.gradient_checkpointing, amr_dim=args.amr_dim,
        use_amr=not args.no_amr, use_agf=not args.no_agf,
    )
    model = CompleteDAGF(**model_kwargs).to(device)
    regularized, unregularized = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (regularized if parameter.ndim > 1 else unregularized).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": regularized, "weight_decay": args.weight_decay},
        {"params": unregularized, "weight_decay": 0.0},
    ], lr=args.bert_lr, eps=args.adam_epsilon)
    if restored:
        model.load_state_dict(restored["model_state"], strict=True)
        optimizer.load_state_dict(restored["optimizer_state"])
        for state in optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value) and name != "step":
                    state[name] = value.to(device)
        _restore_rng(restored["rng_state"])
        LOGGER.info("Resuming epoch=%d next_batch=%d step=%d", restored["next_epoch"] + 1,
                    restored["next_batch"], restored["global_step"])
    checkpoints = Checkpoints(run_dir, args, model, optimizer, vocabulary, model_kwargs, restored)
    evaluation_loader = _loader(evaluation_set, args, evaluation=True)
    _train(model, optimizer, trainset, evaluation_loader, args, device,
           checkpoints, restored, evaluation_split)
    if args.val_file and (run_dir / "best_acc.pt").is_file():
        selected = torch.load(run_dir / "best_acc.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(selected["model_state"], strict=True)
        summary, predictions = _evaluate(model, _loader(testset, args, evaluation=True), device)
        summary["selected_checkpoint"] = "best_acc.pt"
        _atomic_json(run_dir / "final_test.json", summary)
        _atomic_json(run_dir / "final_test_predictions.json", predictions)
        LOGGER.info("Final held-out test: %s", summary)
    LOGGER.info("All outputs kept in %s; legacy baseline files were not changed.", run_dir)
