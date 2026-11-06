"""Sentence-level ABSA evaluation without opinion-span or counterfactual fiction.

Opinion-distance and sharing buckets use a declared lexicon nearest-word proxy.
They are diagnostics, not opinion-extraction scores or ground-truth span buckets.
Gold polarities are used only to score predictions and stratify reports.
"""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import time

import torch
from torch.utils.data import DataLoader

from .data import SentenceABSADataset, sentence_collate


POLARITY_IDS = {"positive": 0, "negative": 1, "neutral": 2}
ID_POLARITIES = {value: key for key, value in POLARITY_IDS.items()}


def model_inputs(batch, device):
    """Keep gold labels outside every inference call, including graph creation."""
    return {
        key: value.to(device)
        for key, value in batch.items()
        if torch.is_tensor(value) and key not in {"labels", "polarity", "sentence_index"}
    }


def classification_metrics(gold, predicted):
    if len(gold) != len(predicted):
        raise ValueError("Gold and prediction counts differ")
    if not gold:
        return {"count": 0, "accuracy": None, "macro_f1": None}
    confusion = [[0] * 3 for _ in range(3)]
    for target, prediction in zip(gold, predicted):
        if target not in ID_POLARITIES or prediction not in ID_POLARITIES:
            raise ValueError("Sentiment labels must be 0, 1 or 2")
        confusion[target][prediction] += 1
    f1 = []
    for label in range(3):
        tp = confusion[label][label]
        denominator = sum(confusion[label]) + sum(row[label] for row in confusion)
        f1.append(2.0 * tp / denominator if denominator else 0.0)
    return {
        "count": len(gold),
        "accuracy": sum(t == p for t, p in zip(gold, predicted)) / len(gold),
        "macro_f1": sum(f1) / 3.0,
        "confusion_matrix": confusion,
    }


def _span(aspect):
    return int(aspect["from"]), int(aspect["to"])


def _nearest_weak_words(metadata):
    candidates = sorted(set(int(index) for index in metadata.get("weak_opinion_indices", [])))
    nearest, distances = [], []
    for aspect in metadata["aspects"]:
        start, end = _span(aspect)
        if not candidates:
            nearest.append([])
            distances.append(None)
            continue
        by_word = {
            index: min(abs(index - aspect_word) for aspect_word in range(start, end))
            for index in candidates
        }
        distance = min(by_word.values())
        nearest.append([index for index in candidates if by_word[index] == distance])
        distances.append(distance)
    counts = Counter(index for indices in nearest for index in indices)
    sharing = [any(counts[index] >= 2 for index in indices) for indices in nearest]
    return nearest, distances, sharing


def _bucket_keys(metadata, aspect_index, distances, sharing):
    count = len(metadata["aspects"])
    labels = [aspect.get("polarity") for aspect in metadata["aspects"]]
    mixed = "unavailable" if any(label is None for label in labels) else (
        "mixed" if len(set(labels)) > 1 else "same"
    )
    distance = distances[aspect_index]
    distance_bucket = "unavailable" if distance is None else (
        "0-2" if distance <= 2 else "3-5" if distance <= 5 else "6+"
    )
    shared_bucket = "unavailable" if distance is None else (
        "shared_proxy" if sharing[aspect_index] else "not_shared_proxy"
    )
    return {
        "aspect_count": str(count) if count < 4 else "4+",
        "sentence_polarity": mixed,
        "opinion_distance_weak_proxy": distance_bucket,
        "shared_opinion_weak_proxy": shared_bucket,
    }


def load_evidence_annotations(source, dataset):
    """Validate externally supplied, human-verified word-span annotations.

    ``source`` is a JSON path, a list of annotation records, or the mapping
    returned by this function. Verification is asserted by the supplier; this
    function checks indices and provenance fields, not annotation truth.
    """
    if source is None:
        return {}
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8") as stream:
            annotations = json.load(stream)
    elif isinstance(source, dict):
        annotations = list(source.values())
    else:
        annotations = source
    if not isinstance(annotations, list):
        raise ValueError("Evidence annotations must be a JSON list")
    sentences = {int(example["sentence_index"]): example for example in dataset.examples}
    result = {}
    for index, annotation in enumerate(annotations):
        location = "evidence annotation %d" % index
        if not isinstance(annotation, dict) or annotation.get("verified") is not True:
            raise ValueError(location + " requires verified=true")
        note = annotation.get("verification_note")
        if not isinstance(note, str) or not note.strip():
            raise ValueError(location + " requires a nonempty verification_note")
        sentence_index, aspect_index = annotation.get("sentence_index"), annotation.get("aspect_index")
        if not isinstance(sentence_index, int) or isinstance(sentence_index, bool) or sentence_index not in sentences:
            raise ValueError(location + " has an invalid sentence_index")
        sentence = sentences[sentence_index]
        if not isinstance(aspect_index, int) or isinstance(aspect_index, bool) or not (
            0 <= aspect_index < len(sentence["aspects"])
        ):
            raise ValueError(location + " has an invalid aspect_index")
        key = (sentence_index, aspect_index)
        if key in result:
            raise ValueError(location + " duplicates a sentence/aspect annotation")
        spans = annotation.get("opinion_spans")
        if not isinstance(spans, list):
            raise ValueError(location + " requires an opinion_spans list")
        checked, seen = [], set()
        for span in spans:
            if not isinstance(span, (list, tuple)) or len(span) != 2 or not all(
                isinstance(value, int) and not isinstance(value, bool) for value in span
            ):
                raise ValueError(location + " spans must be [word_from, word_to] integer intervals")
            start, end = span
            if not 0 <= start < end <= len(sentence["tokens"]):
                raise ValueError(location + " has an opinion span outside the sentence")
            if (start, end) in seen:
                raise ValueError(location + " repeats an opinion span")
            seen.add((start, end))
            checked.append([start, end])
        result[key] = {
            "sentence_index": sentence_index, "aspect_index": aspect_index,
            "opinion_spans": checked, "verified": True, "verification_note": note,
        }
    return result


def _verified_bucket_keys(metadata, aspect_index, annotations):
    sentence_index = int(metadata["sentence_index"])
    annotation = annotations.get((sentence_index, aspect_index))
    if annotation is None:
        return {"opinion_distance_verified": "unavailable", "shared_opinion_verified": "unavailable"}
    words = {word for start, end in annotation["opinion_spans"] for word in range(start, end)}
    if not words:
        return {"opinion_distance_verified": "no_opinion_span", "shared_opinion_verified": "not_shared_verified"}
    start, end = _span(metadata["aspects"][aspect_index])
    distance = min(abs(opinion - aspect) for opinion in words for aspect in range(start, end))
    distance_key = "0-2" if distance <= 2 else "3-5" if distance <= 5 else "6+"
    complete, overlap = True, False
    for other in range(len(metadata["aspects"])):
        if other == aspect_index:
            continue
        other_annotation = annotations.get((sentence_index, other))
        if other_annotation is None:
            complete = False
            continue
        other_words = {word for left, right in other_annotation["opinion_spans"] for word in range(left, right)}
        overlap = overlap or bool(words & other_words)
    shared_key = "shared_verified" if overlap else "not_shared_verified" if complete else "unavailable"
    return {"opinion_distance_verified": distance_key, "shared_opinion_verified": shared_key}


def _route_rows(output, batch, batch_index, aspect_index, metadata, tokenizer):
    word_ids = batch["word_ids"][batch_index].tolist()
    token_ids = batch["text_ids"][batch_index].tolist()
    subwords = tokenizer.convert_ids_to_tokens(token_ids)
    rows = []
    for position, word_id in enumerate(word_ids):
        if word_id < 0:
            continue
        row = {
            "position": position,
            "word_index": word_id,
            "word": metadata["tokens"][word_id],
            "subword": subwords[position],
            "route": float(output["R"][batch_index, aspect_index, position].item()),
            "shared": float(output["R_shared"][batch_index, aspect_index, position].item()),
        }
        if "R_exclusive" in output:
            row["exclusive"] = float(output["R_exclusive"][batch_index, aspect_index, position].item())
        for key in ("background", "shared_channel"):
            if key in output:
                row[key] = float(output[key][batch_index, position].item())
        if "membership" in output:
            row["shared_membership"] = float(output["membership"][batch_index, aspect_index, position].item())
        rows.append(row)
    return rows


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


def _validate_pair(pair, pair_index):
    location = "counterfactual pair %d" % pair_index
    if not isinstance(pair, dict) or pair.get("verified") is not True:
        raise ValueError(location + " requires verified=true")
    if not isinstance(pair.get("verification_note"), str) or not pair["verification_note"].strip():
        raise ValueError(location + " requires a nonempty verification_note")
    before, after = pair.get("before"), pair.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError(location + " requires before and after sentence records")
    if not isinstance(before.get("aspects"), list) or not isinstance(after.get("aspects"), list):
        raise ValueError(location + " requires aspect lists on both sentences")
    if before.get("token") == after.get("token"):
        raise ValueError(location + " contains no textual intervention")
    changed = pair.get("changed_aspects")
    links = pair.get("unchanged_aspect_pairs")
    if not isinstance(changed, list) or not changed or not isinstance(links, list) or not links:
        raise ValueError(location + " requires changed_aspects and unchanged_aspect_pairs")

    def valid_index(index, length):
        return isinstance(index, int) and not isinstance(index, bool) and 0 <= index < length

    if not all(valid_index(index, len(before["aspects"])) for index in changed):
        raise ValueError(location + " has an invalid changed_aspects index")
    if len(set(changed)) != len(changed):
        raise ValueError(location + " repeats a changed_aspects index")
    used_before, used_after = set(), set()
    for link in links:
        if not isinstance(link, list) or len(link) != 2:
            raise ValueError(location + " unchanged links must be [before_index, after_index]")
        source, target = link
        if not valid_index(source, len(before["aspects"])) or not valid_index(target, len(after["aspects"])):
            raise ValueError(location + " has an out-of-range unchanged aspect mapping")
        if source in changed or source in used_before or target in used_after:
            raise ValueError(location + " changed/unchanged aspects overlap or mappings repeat")
        used_before.add(source)
        used_after.add(target)
        left, right = before["aspects"][source], after["aspects"][target]
        left_start, left_end = _span(left)
        right_start, right_end = _span(right)
        if not (0 <= left_start < left_end <= len(before.get("token", []))) or not (
            0 <= right_start < right_end <= len(after.get("token", []))
        ):
            raise ValueError(location + " has an invalid unchanged aspect span")
        left_text = [word.casefold() for word in before["token"][left_start:left_end]]
        right_text = [word.casefold() for word in after["token"][right_start:right_end]]
        if left_text != right_text:
            raise ValueError(location + " unchanged aspect term differs between sentences")
        if left.get("polarity") not in POLARITY_IDS or left.get("polarity") != right.get("polarity"):
            raise ValueError(location + " unchanged aspects require matching valid evaluation gold polarities")
    return before, after, changed, links


def evaluate_counterfactual(model, path, tokenizer, dataset_kwargs, device):
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
            dataset = SentenceABSADataset(
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
