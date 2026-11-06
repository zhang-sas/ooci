"""Token-aligned syntactic and AMR inputs for the default completed DAGF model.

This module reads the supplied word-aligned AMR JSON directly. It does not
modify the existing dataset files or use the legacy pickled graph caches.
"""

import json
from pathlib import Path
import warnings

import torch
from torch.utils.data import Dataset


DATASET_FOLDERS = {
    "restaurant": "Restaurants_corenlp",
    "laptop": "Laptops_corenlp",
    "twitter": "Tweets_corenlp",
    "mams": "MAMS_corenlp",
}
POLARITY_IDS = {"positive": 0, "negative": 1, "neutral": 2}
AMR_SPECIAL_RELATIONS = ("<pad>", "<unk>", "<self>")


def _read_records(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError("Expected a JSON list of sentence records: %s" % path)
    return records


def _relation_label(edge, path, sentence_index):
    if not isinstance(edge, (list, tuple)) or len(edge) != 3:
        raise ValueError(
            "Unsupported AMR edge in %s sentence %d; expected [source_word, relation, target_word]"
            % (path, sentence_index)
        )
    source, relation, target = edge
    if (
        not isinstance(source, int)
        or isinstance(source, bool)
        or not isinstance(target, int)
        or isinstance(target, bool)
        or not isinstance(relation, str)
        or not relation
    ):
        raise ValueError("AMR endpoints must be integer word indices and relation a nonempty string: %s" % (edge,))
    return relation


def _amr_word_alignment(words, amr_words, path, sentence_index):
    """Align word segmentations through identical non-whitespace characters.

    The supplied corpora contain tokens such as ``2\u00a01/2`` which are one
    dependency token but two AMR tokens. AMR endpoints index amr_tokens, so
    checking equal token counts or reusing those indices would be incorrect.
    Each AMR token maps to every original word whose character span overlaps.
    No fuzzy text matching or changes to the source records are performed.
    """
    if not isinstance(amr_words, list) or not amr_words:
        raise ValueError("amr_tokens must be a nonempty list: %s sentence %d" % (path, sentence_index))

    def spans(tokens, field):
        text_parts, intervals = [], []
        cursor = 0
        for token in tokens:
            if not isinstance(token, str):
                raise ValueError("%s must contain strings: %s sentence %d" % (field, path, sentence_index))
            normalized = "".join(character for character in token.casefold() if not character.isspace())
            if not normalized:
                raise ValueError("Whitespace-only token in %s: %s sentence %d" % (field, path, sentence_index))
            text_parts.append(normalized)
            intervals.append((cursor, cursor + len(normalized)))
            cursor += len(normalized)
        return "".join(text_parts), intervals

    original_text, original_spans = spans(words, "token")
    amr_text, amr_spans = spans(amr_words, "amr_tokens")
    if original_text != amr_text:
        raise ValueError(
            "Unsupported AMR alignment in %s sentence %d: token and amr_tokens differ beyond casing/whitespace/token boundaries. "
            "Supply matching word-aligned text rather than guessing concept-node indices."
            % (path, sentence_index)
        )

    alignment = []
    first_original = 0
    for start, end in amr_spans:
        while first_original < len(original_spans) and original_spans[first_original][1] <= start:
            first_original += 1
        mapped = []
        original_index = first_original
        while original_index < len(original_spans) and original_spans[original_index][0] < end:
            original_start, original_end = original_spans[original_index]
            if max(start, original_start) < min(end, original_end):
                mapped.append(original_index)
            original_index += 1
        if not mapped:
            raise ValueError("AMR token has no original-word span: %s sentence %d" % (path, sentence_index))
        alignment.append(mapped)
    return alignment


def build_amr_vocab(train_path):
    """Build relation IDs from training annotations only, including reverse edges."""
    labels = set()
    for sentence_index, record in enumerate(_read_records(train_path)):
        if "amr_edges" not in record:
            raise ValueError("Training data have no amr_edges: %s sentence %d" % (train_path, sentence_index))
        for edge in record["amr_edges"]:
            label = _relation_label(edge, train_path, sentence_index)
            labels.add(label)
            labels.add("reverse:" + label)
    vocab = {label: index for index, label in enumerate(AMR_SPECIAL_RELATIONS)}
    for label in sorted(labels):
        if label not in vocab:
            vocab[label] = len(vocab)
    return vocab


def load_opinion_lexicon(directory):
    """Read Bing Liu's word lists; no sample sentiment labels are consulted."""
    if directory is None:
        return {}
    directory = Path(directory)
    result = {}
    for name, polarity in (("positive-words.txt", 1), ("negative-words.txt", -1)):
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError("Opinion lexicon missing: %s" % path)
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                word = line.strip()
                if word and not word.startswith(";"):
                    result[word.casefold()] = polarity
    return result


class CompleteABSADataset(Dataset):
    """One fixed-length tensor record per annotated sentence-aspect pair.

    ``tokenizer`` must be a fast RoBERTa tokenizer constructed with
    ``add_prefix_space=True``. ``amr_vocab`` must come from the training split,
    and the same vocabulary must be passed to all later splits.
    """

    def __init__(
        self,
        path,
        tokenizer,
        max_length=100,
        amr_vocab=None,
        lexicon_dir=None,
        max_amr_edges=2048,
    ):
        self.path = str(path)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.max_amr_edges = int(max_amr_edges)
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("Use RobertaTokenizerFast(..., add_prefix_space=True) for word_ids alignment")
        if tokenizer.pad_token_id is None or tokenizer.mask_token is None:
            raise ValueError("Tokenizer must define a pad token and an MLM mask token")
        if self.max_length < 16:
            raise ValueError("max_length must be at least 16")
        if self.max_amr_edges < self.max_length:
            raise ValueError("max_amr_edges must reserve at least max_length entries for token self-loops")
        if amr_vocab is None:
            raise ValueError("Pass build_amr_vocab(training_path); do not build relation IDs from test data")
        self.amr_vocab = dict(amr_vocab)
        if [self.amr_vocab.get(name) for name in AMR_SPECIAL_RELATIONS] != [0, 1, 2]:
            raise ValueError("AMR vocabulary requires <pad>=0, <unk>=1 and <self>=2")
        if sorted(self.amr_vocab.values()) != list(range(len(self.amr_vocab))):
            raise ValueError("AMR relation IDs must be contiguous")
        self.lexicon = load_opinion_lexicon(lexicon_dir)
        self.records = _read_records(path)
        self.examples = []
        self.data = []
        self.stats = {
            "sentences": len(self.records),
            "aspects": 0,
            "sentences_without_amr_edges": 0,
            "examples_without_retained_amr_edges": 0,
            "examples_with_truncated_amr_edges": 0,
            "expanded_amr_edges_dropped": 0,
            "unknown_amr_relation_edges": 0,
            "cropped_examples": 0,
            "sentences_with_remapped_amr_tokens": 0,
        }
        for sentence_index, record in enumerate(self.records):
            amr_word_map = self._check_record(record, sentence_index)
            remapped = amr_word_map != [[index] for index in range(len(record["token"]))]
            if remapped:
                self.stats["sentences_with_remapped_amr_tokens"] += 1
            if not record["amr_edges"]:
                self.stats["sentences_without_amr_edges"] += 1
            for aspect_index, aspect in enumerate(record["aspects"]):
                sample_index = len(self.examples)
                metadata = {
                    "sample_index": sample_index,
                    "sentence_index": sentence_index,
                    "aspect_index": aspect_index,
                    "tokens": list(record["token"]),
                    "aspect": " ".join(aspect["term"]),
                    "aspect_from": aspect["from"],
                    "aspect_to": aspect["to"],
                    "polarity": aspect["polarity"],
                    "amr_token_indices_remapped": remapped,
                }
                self.examples.append(metadata)
                self.data.append(self._encode(record, aspect, sample_index, metadata, amr_word_map))
        self.stats["aspects"] = len(self.examples)
        if not self.data:
            raise ValueError("No annotated aspects found in %s" % path)
        if self.stats["examples_without_retained_amr_edges"]:
            warnings.warn(
                "%s: %d aspect examples have no retained annotated AMR edges; their semantic graph contains only word/subword self links."
                % (path, self.stats["examples_without_retained_amr_edges"]),
                RuntimeWarning,
            )
        if self.stats["examples_with_truncated_amr_edges"]:
            warnings.warn(
                "%s: AMR edge budget truncated %d examples. Self-loops and aspect-connected edges take priority; increase --max_amr_edges to retain more."
                % (path, self.stats["examples_with_truncated_amr_edges"]),
                RuntimeWarning,
            )

    def _check_record(self, record, sentence_index):
        required = ("token", "head", "aspects", "amr_edges", "amr_tokens")
        for name in required:
            if name not in record:
                raise ValueError("Missing %s in %s sentence %d" % (name, self.path, sentence_index))
        words = record["token"]
        if not isinstance(words, list) or not words or not all(isinstance(word, str) and word for word in words):
            raise ValueError("token must be a nonempty list of strings: %s sentence %d" % (self.path, sentence_index))
        if len(record["head"]) != len(words):
            raise ValueError("Dependency head/token lengths differ: %s sentence %d" % (self.path, sentence_index))
        amr_words = record["amr_tokens"]
        amr_word_map = _amr_word_alignment(words, amr_words, self.path, sentence_index)
        if not isinstance(record["amr_edges"], list):
            raise ValueError("amr_edges must be a list: %s sentence %d" % (self.path, sentence_index))
        for edge in record["amr_edges"]:
            _relation_label(edge, self.path, sentence_index)
            if not (0 <= edge[0] < len(amr_words) and 0 <= edge[2] < len(amr_words)):
                raise ValueError("AMR endpoint outside amr_tokens list: %s sentence %d edge %s" % (self.path, sentence_index, edge))
        for aspect in record["aspects"]:
            start, end = aspect["from"], aspect["to"]
            if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(words)):
                raise ValueError("Invalid aspect span in %s sentence %d" % (self.path, sentence_index))
            if aspect["polarity"] not in POLARITY_IDS:
                raise ValueError("Unsupported sentiment polarity: %s" % aspect["polarity"])
            if [word.casefold() for word in aspect["term"]] != [word.casefold() for word in words[start:end]]:
                raise ValueError("Aspect term/span mismatch in %s sentence %d" % (self.path, sentence_index))
        for head in record["head"]:
            if not isinstance(head, int) or not 0 <= head <= len(words):
                raise ValueError("Expected 1-based dependency head indices with root=0: %s sentence %d" % (self.path, sentence_index))
        return amr_word_map

    def _word_costs(self, words):
        encoding = self.tokenizer(words, is_split_into_words=True, add_special_tokens=False)
        costs = [0] * len(words)
        for word_id in encoding.word_ids():
            if word_id is not None:
                costs[word_id] += 1
        if any(cost < 1 for cost in costs):
            raise ValueError("Tokenizer produced an empty word; cannot align syntax/AMR in %s" % self.path)
        return costs

    def _crop(self, words, start, end, aspect_words):
        costs = self._word_costs(words)
        aspect_cost = sum(costs[start:end])
        suffix = [self.tokenizer.sep_token, "What", "is", "the", "sentiment", "about"]
        suffix += list(aspect_words) + ["?", "It", "was", self.tokenizer.mask_token]
        suffix_cost = len(self.tokenizer(suffix, is_split_into_words=True, add_special_tokens=False)["input_ids"])
        budget = min(
            self.max_length - aspect_cost - self.tokenizer.num_special_tokens_to_add(pair=True),
            self.max_length - suffix_cost - self.tokenizer.num_special_tokens_to_add(pair=False),
        )
        if aspect_cost > budget:
            raise ValueError(
                "Aspect and its prompt cannot fit max_length=%d in %s; increase max_length. Aspect: %s"
                % (self.max_length, self.path, " ".join(aspect_words))
            )
        left, right, used = start, end, aspect_cost
        left_cost = right_cost = 0
        while left > 0 or right < len(words):
            choices = ("left", "right") if left_cost <= right_cost else ("right", "left")
            added = False
            for side in choices:
                if side == "left" and left > 0 and used + costs[left - 1] <= budget:
                    left -= 1
                    used += costs[left]
                    left_cost += costs[left]
                    added = True
                    break
                if side == "right" and right < len(words) and used + costs[right] <= budget:
                    used += costs[right]
                    right_cost += costs[right]
                    right += 1
                    added = True
                    break
            if not added:
                break
        return left, right, suffix

    def _padded_encoding(self, words, pair=None):
        kwargs = dict(
            is_split_into_words=True,
            add_special_tokens=True,
            padding="max_length",
            truncation=False,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        encoding = self.tokenizer(words, text_pair=pair, **kwargs)
        if len(encoding["input_ids"]) != self.max_length:
            raise ValueError("Encoded sequence exceeded the aspect-preserving crop budget in %s" % self.path)
        return encoding

    def _encode(self, record, aspect, sample_index, metadata, amr_word_map):
        words = record["token"]
        start, end = aspect["from"], aspect["to"]
        aspect_words = words[start:end]
        left, right, prompt_suffix = self._crop(words, start, end, aspect_words)
        cropped_words = words[left:right]
        if left or right != len(words):
            self.stats["cropped_examples"] += 1
        metadata["crop_from"] = left
        metadata["crop_to"] = right
        text = self._padded_encoding(cropped_words, pair=aspect_words)
        prompt = self._padded_encoding(cropped_words + prompt_suffix)
        aspect_encoding = self._padded_encoding(aspect_words)
        word_to_positions = {}
        context_mask = torch.zeros(self.max_length, dtype=torch.bool)
        aspect_mask = torch.zeros(self.max_length, dtype=torch.bool)
        for position, (word_id, sequence_id) in enumerate(zip(text.word_ids(), text.sequence_ids())):
            if word_id is None or sequence_id != 0:
                continue
            original_word = left + word_id
            word_to_positions.setdefault(original_word, []).append(position)
            context_mask[position] = True
            if start <= original_word < end:
                aspect_mask[position] = True
        if not aspect_mask.any():
            raise ValueError("Aspect lost during token alignment: %s example %d" % (self.path, sample_index))
        # Both inputs share exactly <s> + cropped context as their first tokens.
        context_positions = context_mask.nonzero(as_tuple=False).flatten().tolist()
        if any(text["input_ids"][position] != prompt["input_ids"][position] for position in context_positions):
            raise ValueError("Prompt/context token prefixes do not align in %s example %d" % (self.path, sample_index))

        dep_adj = torch.zeros((self.max_length, self.max_length), dtype=torch.float32)
        relation_bias = torch.zeros_like(dep_adj)
        for original_word, positions in word_to_positions.items():
            for source in positions:
                for target in positions:
                    dep_adj[source, target] = 1.0
            parent = record["head"][original_word] - 1
            if parent < 0 or parent not in word_to_positions:
                continue
            lexical = words[original_word].casefold() in self.lexicon or words[parent].casefold() in self.lexicon
            for source in positions:
                for target in word_to_positions[parent]:
                    dep_adj[source, target] = dep_adj[target, source] = 1.0
                    if lexical:
                        relation_bias[source, target] = relation_bias[target, source] = 1.0
        relation_bias += dep_adj

        self_relation = self.amr_vocab["<self>"]
        unknown_relation = self.amr_vocab["<unk>"]
        aspect_positions = set(aspect_mask.nonzero(as_tuple=False).flatten().tolist())
        candidates = set()
        # A word expanded into several subwords keeps an internal identity link.
        for positions in word_to_positions.values():
            for source in positions:
                for target in positions:
                    candidates.add((source, target, self_relation))
        retained_annotated_edges = 0
        unknown_edges = 0
        for source_amr, label, target_amr in record["amr_edges"]:
            source_words = [word for word in amr_word_map[source_amr] if word in word_to_positions]
            target_words = [word for word in amr_word_map[target_amr] if word in word_to_positions]
            if not source_words or not target_words:
                continue
            retained_annotated_edges += 1
            forward_id = self.amr_vocab.get(label, unknown_relation)
            reverse_id = self.amr_vocab.get("reverse:" + label, unknown_relation)
            if forward_id == unknown_relation:
                unknown_edges += 1
            for source_word in source_words:
                for target_word in target_words:
                    for source in word_to_positions[source_word]:
                        for target in word_to_positions[target_word]:
                            candidates.add((source, target, forward_id))
                            candidates.add((target, source, reverse_id))

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
        if dropped:
            self.stats["examples_with_truncated_amr_edges"] += 1
            self.stats["expanded_amr_edges_dropped"] += dropped
        if not retained_annotated_edges:
            self.stats["examples_without_retained_amr_edges"] += 1
        self.stats["unknown_amr_relation_edges"] += unknown_edges
        metadata["retained_amr_word_edges"] = retained_annotated_edges
        metadata["expanded_amr_edges_dropped"] = dropped
        amr_src = torch.zeros(self.max_amr_edges, dtype=torch.long)
        amr_dst = torch.zeros(self.max_amr_edges, dtype=torch.long)
        amr_rel = torch.zeros(self.max_amr_edges, dtype=torch.long)
        amr_edge_mask = torch.zeros(self.max_amr_edges, dtype=torch.bool)
        amr_adjacency = torch.zeros_like(dep_adj)
        for edge_index, (source, target, relation) in enumerate(selected):
            amr_src[edge_index] = source
            amr_dst[edge_index] = target
            amr_rel[edge_index] = relation
            amr_edge_mask[edge_index] = True
            amr_adjacency[source, target] = 1.0

        return {
            "text_ids": torch.tensor(text["input_ids"], dtype=torch.long),
            "text_attention_mask": torch.tensor(text["attention_mask"], dtype=torch.long),
            "prompt_ids": torch.tensor(prompt["input_ids"], dtype=torch.long),
            "prompt_attention_mask": torch.tensor(prompt["attention_mask"], dtype=torch.long),
            "aspect_ids": torch.tensor(aspect_encoding["input_ids"], dtype=torch.long),
            "aspect_attention_mask": torch.tensor(aspect_encoding["attention_mask"], dtype=torch.long),
            "context_mask": context_mask,
            "aspect_mask": aspect_mask,
            "dep_adj": dep_adj,
            "relation_bias": relation_bias,
            "amr_adjacency": amr_adjacency,
            "amr_src": amr_src,
            "amr_dst": amr_dst,
            "amr_rel": amr_rel,
            "amr_edge_mask": amr_edge_mask,
            "amr_edge_count": torch.tensor(len(selected), dtype=torch.long),
            "amr_original_edge_count": torch.tensor(retained_annotated_edges, dtype=torch.long),
            "polarity": torch.tensor(POLARITY_IDS[aspect["polarity"]], dtype=torch.long),
            "sample_index": torch.tensor(sample_index, dtype=torch.long),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
