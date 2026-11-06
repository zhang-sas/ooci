"""Original metrics/protocol with dependency-aware collation and dataset factories."""
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import time
import torch
from torch.utils.data import DataLoader

from ..typed_syntax.data import TypedSentenceDataset, TypedTwitterDataset, typed_collate as sentence_collate
from ..v1_reference.evaluation import (
    model_inputs, classification_metrics, POLARITY_IDS, ID_POLARITIES,
    _span, _nearest_weak_words, _bucket_keys, load_evidence_annotations,
    _verified_bucket_keys, _route_rows, _validate_pair,
)


def _dataset_class(name):
    return TypedTwitterDataset if name == "twitter" else TypedSentenceDataset


@torch.inference_mode()
def evaluate(model, dataset, device, batch_size=8, num_workers=0, include_routes=False,
             evidence_annotations=None):
    """Return ``(report, sentence_predictions)`` with aspect-level metrics.

    Macro-F1 always averages positive/negative/neutral, including empty classes.
    Unlabelled aspects can be predicted but are excluded from metric counts.
    """
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    device = torch.device(device)
    annotations = load_evidence_annotations(evidence_annotations, dataset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=sentence_collate)
    gold_all, predicted_all, predictions = [], [], []
    buckets = defaultdict(lambda: defaultdict(lambda: ([], [])))
    cursor = 0
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            output = model(model_inputs(batch, device))
            logits = output["logits"].detach().cpu()
            probabilities = torch.softmax(logits, dim=-1)
            valid = batch["aspect_valid"].bool()
            if logits.shape[:2] != valid.shape or logits.size(-1) != 3:
                raise ValueError("Expected logits [B,A,3] matching aspect_valid")
            for batch_index in range(valid.size(0)):
                metadata = dataset.examples[cursor]
                cursor += 1
                nearest, distances, sharing = _nearest_weak_words(metadata)
                sentence = {
                    "sentence_index": int(metadata["sentence_index"]),
                    "tokens": metadata["tokens"],
                    "aspects": [],
                }
                count = int(valid[batch_index].sum().item())
                if count != len(metadata["aspects"]):
                    raise ValueError("Aspect metadata and model batch disagree")
                for aspect_index, aspect in enumerate(metadata["aspects"]):
                    if not valid[batch_index, aspect_index]:
                        raise ValueError("Valid aspects must precede padded slots")
                    prediction = int(probabilities[batch_index, aspect_index].argmax().item())
                    target = aspect.get("polarity")
                    target_id = POLARITY_IDS.get(target)
                    start, end = _span(aspect)
                    term = aspect.get("term", metadata["tokens"][start:end])
                    record = {
                        "aspect_index": int(aspect.get("aspect_index", aspect_index)),
                        "from": start,
                        "to": end,
                        "term": term,
                        "gold": target,
                        "gold_id": target_id,
                        "pred": ID_POLARITIES[prediction],
                        "pred_id": prediction,
                        "confidence": float(probabilities[batch_index, aspect_index, prediction].item()),
                        "probabilities": probabilities[batch_index, aspect_index].tolist(),
                        "weak_nearest_opinion_indices": nearest[aspect_index],
                    }
                    if "depth_weights" in output:
                        record["depth_weights"] = output["depth_weights"][batch_index, aspect_index].detach().cpu().tolist()
                    if include_routes:
                        record["routing"] = _route_rows(output, batch, batch_index,
                                                         aspect_index, metadata, dataset.tokenizer)
                    sentence["aspects"].append(record)
                    if target_id is None:
                        continue
                    gold_all.append(target_id)
                    predicted_all.append(prediction)
                    keys = _bucket_keys(metadata, aspect_index, distances, sharing)
                    if annotations:
                        keys.update(_verified_bucket_keys(metadata, aspect_index, annotations))
                    for family, key in keys.items():
                        bucket_gold, bucket_predictions = buckets[family][key]
                        bucket_gold.append(target_id)
                        bucket_predictions.append(prediction)
                predictions.append(sentence)
    finally:
        model.train(was_training)
    report = classification_metrics(gold_all, predicted_all)
    report.update({
        "sentences": len(predictions),
        "aspects_predicted": sum(len(sentence["aspects"]) for sentence in predictions),
        "metric_unit": "aspect",
        "macro_f1_labels": ["positive", "negative", "neutral"],
        "buckets": {
            family: {key: classification_metrics(*values) for key, values in sorted(groups.items())}
            for family, groups in sorted(buckets.items())
        },
        "bucket_provenance": {
            "aspect_count": "annotated aspect spans",
            "sentence_polarity": "evaluation gold labels only; never model inputs or edges",
            "opinion_distance_weak_proxy": "minimum word-index distance to lexicon hits; no gold opinion spans",
            "shared_opinion_weak_proxy": "at least two aspects select the same nearest lexicon hit; all distance ties included",
        },
        "gold_opinion_span_evaluation": {
            "status": "evaluated" if annotations else "not_evaluated",
            "annotated_aspects": len(annotations),
            "reason": "Externally supplied verification assertions and word spans; supplier is responsible for annotation truth." if annotations else
                      "No externally verified opinion spans supplied; weak proxy buckets are reported separately.",
            "sharing_rule": "overlap of annotated opinion word spans across aspects; missing other-aspect coverage makes negative sharing unavailable",
        },
    })
    if annotations:
        report["bucket_provenance"].update({
            "opinion_distance_verified": "minimum word-index distance to externally verified opinion spans",
            "shared_opinion_verified": "overlapping externally verified word-span evidence across aspects; independent of lexicon and model routes",
        })
    report["base_model"] = "frozen_v1"
    report["additive_relation_messages"] = False
    report["relation_gate_diagnostics"] = model.gate_diagnostics()
    return report, predictions


def _percentile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@torch.inference_mode()
def benchmark_latency(model, dataset, device, warmup=2, repeats=1, max_sentences=100):
    """Batch-one forward latency; includes graphs, routing, fusion and classifier.

    Data loading, tokenization and host-to-device transfer are outside the timer.
    CUDA work is synchronized both before and after each timed forward.
    """
    if warmup < 0 or repeats < 1 or max_sentences < 1:
        raise ValueError("Require warmup >= 0, repeats >= 1 and max_sentences >= 1")
    size = min(len(dataset), max_sentences)
    if size == 0:
        return {"status": "not_evaluated", "reason": "Empty dataset"}
    device = torch.device(device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    was_training = model.training
    model.eval()
    elapsed, by_count = [], defaultdict(list)
    try:
        first = model_inputs(sentence_collate([dataset[0]]), device)
        for _ in range(warmup):
            model(first)
        synchronize()
        for index in range(size):
            batch = sentence_collate([dataset[index]])
            inputs = model_inputs(batch, device)
            count = int(batch["aspect_valid"].sum().item())
            for _ in range(repeats):
                synchronize()
                start = time.perf_counter()
                model(inputs)
                synchronize()
                duration = (time.perf_counter() - start) * 1000.0
                elapsed.append(duration)
                by_count[str(count) if count < 4 else "4+"].append(duration)
    finally:
        model.train(was_training)
    return {
        "status": "evaluated",
        "device": str(device),
        "batch_size": 1,
        "sentences": size,
        "repeats": repeats,
        "timed_forwards": len(elapsed),
        "warmup_forwards": warmup,
        "mean_ms_per_sentence": statistics.mean(elapsed),
        "p50_ms_per_sentence": _percentile(elapsed, 0.5),
        "p95_ms_per_sentence": _percentile(elapsed, 0.95),
        "by_aspect_count": {
            count: {"timed_forwards": len(values), "mean_ms_per_sentence": statistics.mean(values)}
            for count, values in sorted(by_count.items())
        },
        "scope": "preencoded batch-one model forward; excludes tokenization, data loading and host-to-device transfer",
        "encoding": "one shared text-encoder forward and one shared DAGF MLM forward per sentence; no per-aspect encoder forward",
    }


def evaluate_counterfactual(model, path, tokenizer, dataset_kwargs, device, dataset_name="restaurant"):
    """Score independently verified interventions with stable aspect mappings.

    The unconditional flip rate and the conditional newly-wrong rate have distinct
    denominators. There is no estimate from synthetic or unverified pairs.
    """
    if path is None:
        return {"status": "not_evaluated", "reason": "No independently verified counterfactual pairs supplied"}
    with Path(path).open("r", encoding="utf-8") as stream:
        pairs = json.load(stream)
    if not isinstance(pairs, list):
        raise ValueError("Counterfactual input must be a JSON list of verified pairs")
    if not pairs:
        return {"status": "not_evaluated", "reason": "Counterfactual pair file is empty"}
    validated = [_validate_pair(pair, index) for index, pair in enumerate(pairs)]
    unchanged_count = flip_count = before_correct_count = new_error_count = 0
    details = []
    # Each record is encoded independently to preserve caller-supplied indices,
    # even if the normal dataset coalesces duplicate sentence records.
    kwargs = dict(dataset_kwargs)
    kwargs.pop("tokenizer", None)
    kwargs["require_labels"] = True
    for pair_index, (pair, values) in enumerate(zip(pairs, validated)):
        before, after, changed, links = values
        predicted_sides = []
        for side, record in (("before", before), ("after", after)):
            dataset = _dataset_class(dataset_name)(
                "<counterfactual:%d:%s>" % (pair_index, side), tokenizer=tokenizer,
                records=[record], **kwargs,
            )
            _, predictions = evaluate(model, dataset, device, batch_size=1)
            if len(predictions) != 1:
                raise ValueError("Counterfactual side must encode exactly one sentence")
            predicted_sides.append(predictions[0]["aspects"])
        before_predictions, after_predictions = predicted_sides
        observations = []
        for source, target in links:
            left, right = before_predictions[source], after_predictions[target]
            target_id = POLARITY_IDS[before["aspects"][source]["polarity"]]
            flipped = left["pred_id"] != right["pred_id"]
            before_correct = left["pred_id"] == target_id
            newly_wrong = before_correct and right["pred_id"] != target_id
            unchanged_count += 1
            flip_count += int(flipped)
            before_correct_count += int(before_correct)
            new_error_count += int(newly_wrong)
            observations.append({
                "before_aspect_index": source, "after_aspect_index": target,
                "gold": ID_POLARITIES[target_id], "before_prediction": left["pred"],
                "after_prediction": right["pred"], "flipped": flipped,
                "before_correct": before_correct, "newly_wrong": newly_wrong,
            })
        details.append({"id": pair.get("id", pair_index), "changed_aspects": changed,
                        "verification_note": pair["verification_note"], "unchanged_aspects": observations})
    return {
        "status": "evaluated", "verified_pairs": len(pairs),
        "unchanged_aspect_count": unchanged_count, "flip_count": flip_count,
        "other_aspect_flip_rate": flip_count / unchanged_count if unchanged_count else None,
        "before_correct_unchanged_count": before_correct_count,
        "new_error_count": new_error_count,
        "collateral_error_rate": new_error_count / before_correct_count if before_correct_count else None,
        "collateral_error_denominator": "unchanged aspects correctly predicted before the intervention",
        "new_error_rate_over_all_unchanged": new_error_count / unchanged_count if unchanged_count else None,
        "details": details,
    }
