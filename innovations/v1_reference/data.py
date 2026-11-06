"""Sentence-level, label-free graph inputs for aspect evidence routing.

The original data files and baseline loader are never changed. Opinion lexicon
hits are weak diagnostic candidates, not annotated opinion spans. Sentiment
labels are returned separately and never enter the graphs or routing priors.
"""

import copy
import json
from collections import OrderedDict
import warnings

import torch
from torch.utils.data import Dataset

from .baseline import load_baseline_module

_baseline_data = load_baseline_module("data")
AMR_SPECIAL_RELATIONS = _baseline_data.AMR_SPECIAL_RELATIONS
DATASET_FOLDERS = _baseline_data.DATASET_FOLDERS
POLARITY_IDS = _baseline_data.POLARITY_IDS
_amr_word_alignment = _baseline_data._amr_word_alignment
_read_records = _baseline_data._read_records
_relation_label = _baseline_data._relation_label
build_amr_vocab = _baseline_data.build_amr_vocab
load_opinion_lexicon = _baseline_data.load_opinion_lexicon


CONTRAST_MARKERS = frozenset(("but", "however", "although", "though", "whereas", "yet", "nevertheless"))
COMPARISON_MARKERS = frozenset(("than", "versus", "vs", "vs.", "compared", "unlike"))
COORDINATION_MARKERS = frozenset(("and", "or", "also", "both"))


def _merge_sentences(records, path):
    """Merge repeated sentence records without inventing or losing aspects."""
    grouped = OrderedDict()
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("Sentence record must be an object: %s record %d" % (path, record_index))
        for key in ("token", "head", "amr_tokens", "amr_edges", "aspects"):
            if key not in record:
                raise ValueError("Missing %s in %s record %d" % (key, path, record_index))
        # Edges are sorted only for the grouping key. Supplied graph order and
        # all other source fields are retained in the first sentence record.
        key = json.dumps(
            [record["token"], record["head"], record["amr_tokens"],
             sorted(record["amr_edges"], key=lambda edge: json.dumps(edge, sort_keys=True))],
            ensure_ascii=False, sort_keys=True,
        )
        if key not in grouped:
            sentence = copy.deepcopy(record)
            sentence["aspects"] = []
            grouped[key] = (sentence, [], OrderedDict())
        sentence, source_indices, seen = grouped[key]
        source_indices.append(record_index)
        if not isinstance(record["aspects"], list):
            raise ValueError("aspects must be a list: %s record %d" % (path, record_index))
        for aspect in record["aspects"]:
            if not isinstance(aspect, dict):
                raise ValueError("Each aspect must be an object: %s record %d" % (path, record_index))
            if not all(field in aspect for field in ("from", "to", "term")):
                raise ValueError("Aspect requires from/to/term: %s record %d" % (path, record_index))
            aspect_key = (aspect["from"], aspect["to"])
            if aspect_key in seen:
                previous = seen[aspect_key]
                old_label, new_label = previous.get("polarity"), aspect.get("polarity")
                if old_label is not None and new_label is not None and old_label != new_label:
                    raise ValueError(
                        "Conflicting duplicate aspect polarities in %s records %s at span %s"
                        % (path, source_indices, aspect_key)
                    )
                if old_label is None and new_label is not None:
                    previous["polarity"] = new_label
                if previous["term"] != aspect["term"]:
                    raise ValueError("Conflicting duplicate aspect terms: %s span %s" % (path, aspect_key))
                continue
            copied_aspect = copy.deepcopy(aspect)
            seen[aspect_key] = copied_aspect
            sentence["aspects"].append(copied_aspect)
    return [(sentence, indices) for sentence, indices, _ in grouped.values()]


class SentenceABSADataset(Dataset):
    """One example per sentence, retaining its entire variable-size aspect list.

    No cropping is performed: both the full text and its fixed generic prompt
    must fit ``max_length``. Long sentences raise an actionable error rather
    than silently removing aspects or their possible evidence.
    """

    def __init__(
        self, path, tokenizer, max_length=160, amr_vocab=None,
        lexicon_dir=None, max_amr_edges=2048, require_labels=True, records=None,
    ):
        self.path = str(path)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.max_amr_edges = int(max_amr_edges)
        self.require_labels = bool(require_labels)
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("Use RobertaTokenizerFast(..., add_prefix_space=True) for word alignment")
        if any(getattr(tokenizer, name, None) is None for name in ("pad_token_id", "mask_token", "sep_token")):
            raise ValueError("Tokenizer must define pad, mask and separator tokens")
        if self.max_length < 16:
            raise ValueError("max_length must be at least 16")
        if self.max_amr_edges < self.max_length:
            raise ValueError("max_amr_edges must reserve at least max_length token self-loops")
        if amr_vocab is None:
            raise ValueError("Supply build_amr_vocab(training_path); do not build vocabulary from validation/test")
        self.amr_vocab = dict(amr_vocab)
        if [self.amr_vocab.get(name) for name in AMR_SPECIAL_RELATIONS] != [0, 1, 2]:
            raise ValueError("AMR vocabulary requires <pad>=0, <unk>=1 and <self>=2")
        if sorted(self.amr_vocab.values()) != list(range(len(self.amr_vocab))):
            raise ValueError("AMR relation IDs must be contiguous")
        self.lexicon = load_opinion_lexicon(lexicon_dir)
        raw_records = _read_records(path) if records is None else copy.deepcopy(records)
        if not isinstance(raw_records, list):
            raise ValueError("records must be a list of sentence objects")
        grouped = _merge_sentences(raw_records, self.path)
        self.records = [record for record, _ in grouped]
        self.examples, self.data = [], []
        self.stats = {
            "source_records": len(raw_records), "sentences": len(grouped),
            "merged_source_records": len(raw_records) - len(grouped), "aspects": 0,
            "sentences_without_amr_edges": 0,
            "sentences_with_remapped_amr_tokens": 0,
            "sentences_with_truncated_amr_edges": 0,
            "expanded_amr_edges_dropped": 0, "unknown_amr_relation_edges": 0,
            "max_encoded_length": 0,
            "lexicon_candidate_source": "word_list_weak_proxy" if self.lexicon else "unavailable",
            "opinion_span_gold_available": False,
        }
        for sentence_index, (record, source_indices) in enumerate(grouped):
            amr_map = self._check_record(record, sentence_index)
            remapped = amr_map != [[index] for index in range(len(record["token"]))]
            self.stats["sentences_with_remapped_amr_tokens"] += int(remapped)
            self.stats["sentences_without_amr_edges"] += int(not record["amr_edges"])
            aspects = []
            for aspect_index, aspect in enumerate(record["aspects"]):
                metadata = copy.deepcopy(aspect)
                metadata["aspect_index"] = aspect_index
                metadata["polarity"] = aspect.get("polarity")
                aspects.append(metadata)
            labels = [aspect["polarity"] for aspect in aspects]
            metadata = {
                "sentence_index": sentence_index,
                "source_record_indices": source_indices,
                "tokens": list(record["token"]), "aspects": aspects,
                "aspect_count": len(aspects),
                "weak_opinion_indices": [i for i, word in enumerate(record["token"]) if word.casefold() in self.lexicon],
                "weak_opinion_source": "word_list_weak_proxy" if self.lexicon else None,
                "has_mixed_polarity": len(set(labels)) > 1 if all(label is not None for label in labels) else None,
                "amr_token_indices_remapped": remapped,
            }
            self.examples.append(metadata)
            self.data.append(self._encode(record, sentence_index, metadata, amr_map))
            self.stats["aspects"] += len(aspects)
        if not self.data:
            raise ValueError("No sentences found in %s" % self.path)
        if self.stats["sentences_with_truncated_amr_edges"]:
            warnings.warn(
                "%s: AMR budget truncated %d sentences (%d expanded edges); increase max_amr_edges."
                % (self.path, self.stats["sentences_with_truncated_amr_edges"], self.stats["expanded_amr_edges_dropped"]),
                RuntimeWarning,
            )

    def _check_record(self, record, sentence_index):
        words = record["token"]
        location = "%s sentence %d" % (self.path, sentence_index)
        if not isinstance(words, list) or not words or not all(isinstance(word, str) and word.strip() for word in words):
            raise ValueError("token must be a nonempty list of nonblank strings: " + location)
        if not isinstance(record["head"], list) or len(record["head"]) != len(words):
            raise ValueError("Dependency head/token lengths differ: " + location)
        for head in record["head"]:
            if isinstance(head, bool) or not isinstance(head, int) or not 0 <= head <= len(words):
                raise ValueError("Expected 1-based dependency heads, with root=0: " + location)
        amr_map = _amr_word_alignment(words, record["amr_tokens"], self.path, sentence_index)
        if not isinstance(record["amr_edges"], list):
            raise ValueError("amr_edges must be a list: " + location)
        for edge in record["amr_edges"]:
            _relation_label(edge, self.path, sentence_index)
            if not (0 <= edge[0] < len(amr_map) and 0 <= edge[2] < len(amr_map)):
                raise ValueError("AMR endpoint outside amr_tokens: " + location)
        if not record["aspects"]:
            raise ValueError("Sentence must contain at least one annotated aspect: " + location)
        for aspect in record["aspects"]:
            start, end = aspect["from"], aspect["to"]
            if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int)
                    or not isinstance(end, int) or not 0 <= start < end <= len(words)):
                raise ValueError("Invalid aspect span: " + location)
            term = aspect["term"]
            if not isinstance(term, list) or not all(isinstance(word, str) for word in term):
                raise ValueError("Aspect term must be a token list: " + location)
            if [word.casefold() for word in term] != [word.casefold() for word in words[start:end]]:
                raise ValueError("Aspect term/span mismatch: " + location)
            polarity = aspect.get("polarity")
            if polarity is None and not self.require_labels:
                continue
            if polarity not in POLARITY_IDS:
                raise ValueError("Missing or unsupported sentiment polarity: " + location)
        return amr_map

    def _encoding(self, words, sentence_index, field):
        encoding = self.tokenizer(
            words, is_split_into_words=True, add_special_tokens=True,
            padding=False, truncation=False,
            return_attention_mask=True,
        )
        if len(encoding["input_ids"]) > self.max_length:
            raise ValueError(
                "%s sentence %d %s needs at least max_length=%d (configured %d). "
                "Increase max_length; full sentences and all aspects must be preserved."
                % (self.path, sentence_index, field, len(encoding["input_ids"]), self.max_length)
            )
        return encoding

    def _priors(self, words, aspects, word_to_positions, length):
        count = len(aspects)
        route_prior = torch.zeros((count, length), dtype=torch.float32)
        relation = torch.zeros((count, count), dtype=torch.long)
        lower = [word.casefold() for word in words]
        contrast_indices = [i for i, word in enumerate(lower) if word in CONTRAST_MARKERS]
        for aspect_index, aspect in enumerate(aspects):
            start, end = aspect["from"], aspect["to"]
            for word_index, positions in word_to_positions.items():
                distance = max(start - word_index, word_index - end + 1, 0)
                left = min(word_index, start)
                right = max(word_index, end - 1)
                crossings = sum(left < marker < right for marker in contrast_indices)
                # Soft, entirely textual priors. They never prohibit shared
                # opinions and contain no sentiment sign or gold polarity.
                value = -0.05 * distance - 0.5 * crossings
                route_prior[aspect_index, positions] = value
            for other_index, other in enumerate(aspects):
                if aspect_index == other_index:
                    continue
                first, second = sorted((aspect, other), key=lambda item: item["from"])
                between = set(lower[first["to"]:second["from"]])
                if between & CONTRAST_MARKERS:
                    relation[aspect_index, other_index] = 2
                elif between & COMPARISON_MARKERS:
                    relation[aspect_index, other_index] = 3
                elif between & COORDINATION_MARKERS:
                    relation[aspect_index, other_index] = 1
        return route_prior, relation

    def _encode(self, record, sentence_index, metadata, amr_map):
        words, aspects = record["token"], record["aspects"]
        text = self._encoding(words, sentence_index, "text")
        # The same generic prompt is used irrespective of labels/aspect count.
        suffix = [self.tokenizer.sep_token, "What", "opinions", "are", "expressed", "?", self.tokenizer.mask_token]
        prompt = self._encoding(words + suffix, sentence_index, "prompt")
        # Sentence-local padding avoids materializing max_length squared graph
        # tensors for every short sentence. The collator pads to batch length.
        length = max(len(text["input_ids"]), len(prompt["input_ids"]))
        self.stats["max_encoded_length"] = max(self.stats["max_encoded_length"], length)
        context_mask = torch.zeros(length, dtype=torch.bool)
        aspect_mask = torch.zeros((len(aspects), length), dtype=torch.bool)
        word_ids = torch.full((length,), -1, dtype=torch.long)
        word_to_positions = {}
        for position, word_index in enumerate(text.word_ids()):
            if word_index is None:
                continue
            word_to_positions.setdefault(word_index, []).append(position)
            word_ids[position] = word_index
            context_mask[position] = True
            if text["input_ids"][position] != prompt["input_ids"][position]:
                raise ValueError("Prompt/context subword prefixes do not align: %s sentence %d" % (self.path, sentence_index))
            for aspect_index, aspect in enumerate(aspects):
                if aspect["from"] <= word_index < aspect["to"]:
                    aspect_mask[aspect_index, position] = True
        if len(word_to_positions) != len(words) or not aspect_mask.any(dim=1).all():
            raise ValueError("Tokenizer lost a word or aspect: %s sentence %d" % (self.path, sentence_index))
        dep_adj = torch.zeros((length, length), dtype=torch.float32)
        relation_bias = torch.zeros_like(dep_adj)
        for word_index, positions in word_to_positions.items():
            for source in positions:
                for target in positions:
                    dep_adj[source, target] = 1.0
            parent = record["head"][word_index] - 1
            if parent < 0:
                continue
            lexical = words[word_index].casefold() in self.lexicon or words[parent].casefold() in self.lexicon
            for source in positions:
                for target in word_to_positions[parent]:
                    dep_adj[source, target] = dep_adj[target, source] = 1.0
                    if lexical:
                        relation_bias[source, target] = relation_bias[target, source] = 1.0
        relation_bias += dep_adj
        route_prior, aspect_relation = self._priors(words, aspects, word_to_positions, length)
        self_relation, unknown = self.amr_vocab["<self>"], self.amr_vocab["<unk>"]
        candidates = set()
        for positions in word_to_positions.values():
            for source in positions:
                for target in positions:
                    candidates.add((source, target, self_relation))
        unknown_edges = 0
        for source_amr, label, target_amr in record["amr_edges"]:
            forward_id = self.amr_vocab.get(label, unknown)
            reverse_id = self.amr_vocab.get("reverse:" + label, unknown)
            unknown_edges += int(forward_id == unknown)
            for source_word in amr_map[source_amr]:
                for target_word in amr_map[target_amr]:
                    for source in word_to_positions[source_word]:
                        for target in word_to_positions[target_word]:
                            candidates.add((source, target, forward_id))
                            candidates.add((target, source, reverse_id))
        aspect_positions = set(aspect_mask.any(dim=0).nonzero(as_tuple=False).flatten().tolist())

        def priority(edge):
            source, target, relation = edge
            if source == target and relation == self_relation:
                rank = 0
            elif source in aspect_positions or target in aspect_positions:
                rank = 1
            elif relation != self_relation:
                rank = 2
            else:
                rank = 3
            return rank, source, target, relation

        ordered = sorted(candidates, key=priority)
        selected = ordered[:self.max_amr_edges]
        dropped = len(ordered) - len(selected)
        self.stats["sentences_with_truncated_amr_edges"] += int(dropped > 0)
        self.stats["expanded_amr_edges_dropped"] += dropped
        self.stats["unknown_amr_relation_edges"] += unknown_edges
        metadata["retained_amr_word_edges"] = len(record["amr_edges"])
        metadata["expanded_amr_edges_dropped"] = dropped
        amr_src = torch.zeros(self.max_amr_edges, dtype=torch.long)
        amr_dst = torch.zeros_like(amr_src)
        amr_rel = torch.zeros_like(amr_src)
        edge_mask = torch.zeros(self.max_amr_edges, dtype=torch.bool)
        if selected:
            edge_values = torch.tensor(selected, dtype=torch.long).transpose(0, 1)
            edge_count = len(selected)
            amr_src[:edge_count] = edge_values[0]
            amr_dst[:edge_count] = edge_values[1]
            amr_rel[:edge_count] = edge_values[2]
            edge_mask[:edge_count] = True
        def padded(encoding, key):
            fill = self.tokenizer.pad_token_id if key == "input_ids" else 0
            values = encoding[key]
            return torch.tensor(values + [fill] * (length - len(values)), dtype=torch.long)

        return {
            "text_ids": padded(text, "input_ids"),
            "text_attention_mask": padded(text, "attention_mask"),
            "prompt_ids": padded(prompt, "input_ids"),
            "prompt_attention_mask": padded(prompt, "attention_mask"),
            "context_mask": context_mask, "aspect_mask": aspect_mask,
            "aspect_valid": torch.ones(len(aspects), dtype=torch.bool),
            "labels": torch.tensor([POLARITY_IDS.get(aspect.get("polarity"), -100) for aspect in aspects], dtype=torch.long),
            "dep_adj": dep_adj, "relation_bias": relation_bias,
            "amr_src": amr_src, "amr_dst": amr_dst, "amr_rel": amr_rel,
            "amr_edge_mask": edge_mask, "route_prior": route_prior,
            "aspect_relation": aspect_relation, "word_ids": word_ids,
            "sentence_index": torch.tensor(sentence_index, dtype=torch.long),
            "_pad_token_id": self.tokenizer.pad_token_id,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def sentence_collate(examples):
    """Dynamically pad sentence and aspect axes, preserving every aspect."""
    if not examples:
        raise ValueError("Cannot collate an empty sentence batch")
    max_aspects = max(example["aspect_valid"].numel() for example in examples)
    max_length = max(example["text_ids"].numel() for example in examples)
    pad_ids = {example["_pad_token_id"] for example in examples}
    if len(pad_ids) != 1:
        raise ValueError("Cannot collate different tokenizer padding IDs")
    pad_id = next(iter(pad_ids))
    sequence_vectors = {"text_ids", "text_attention_mask", "prompt_ids", "prompt_attention_mask", "context_mask", "word_ids"}
    sequence_matrices = {"dep_adj", "relation_bias"}
    aspect_vectors = {"aspect_valid", "labels"}
    aspect_sequences = {"aspect_mask", "route_prior"}
    result = {}
    for key in examples[0]:
        if key.startswith("_"):
            continue
        if key not in sequence_vectors | sequence_matrices | aspect_vectors | aspect_sequences | {"aspect_relation"}:
            result[key] = torch.stack([example[key] for example in examples])
            continue
        padded = []
        for example in examples:
            value = example[key]
            if key == "aspect_relation":
                target = value.new_zeros((max_aspects, max_aspects))
                target[:value.size(0), :value.size(1)] = value
            elif key in aspect_sequences:
                target = value.new_zeros((max_aspects, max_length))
                target[:value.size(0), :value.size(1)] = value
            elif key in sequence_matrices:
                target = value.new_zeros((max_length, max_length))
                target[:value.size(0), :value.size(1)] = value
            elif key in sequence_vectors:
                fill = pad_id if key in {"text_ids", "prompt_ids"} else (-1 if key == "word_ids" else 0)
                target = value.new_full((max_length,), fill)
                target[:value.size(0)] = value
            else:
                shape = (max_aspects,) + tuple(value.shape[1:])
                target = value.new_full(shape, -100 if key == "labels" else 0)
                target[:value.size(0)] = value
            padded.append(target)
        result[key] = torch.stack(padded)
    return result


__all__ = [
    "SentenceABSADataset", "sentence_collate", "build_amr_vocab",
    "DATASET_FOLDERS", "POLARITY_IDS", "AMR_SPECIAL_RELATIONS",
]
