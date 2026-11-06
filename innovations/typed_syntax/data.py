"""Parser-derived edge attributes on exactly the frozen v1 syntax topology.

The original loaders keep tokenization, graph weights, sentence merging and
labels. This adapter only adds sparse attributes to their existing edges.
Dependency vocabularies must be built from the training file and saved with the
checkpoint; evaluation never expands them. No polarity enters an edge feature.
"""
import copy
import json

import torch

from ..v1_reference.data import SentenceABSADataset, _read_records, sentence_collate
from ..twitter_v1.data import TwitterABSADataset


DEPENDENCY_SPECIAL_RELATIONS = ("<pad>", "<unk>", "<self>", "<intra_word>")
DEP_EDGE_KEYS = frozenset(("dep_src", "dep_dst", "dep_type", "dep_direction", "dep_edge_mask"))
SELF_DIRECTION = 0
FORWARD_DIRECTION = 1  # head -> dependent (message source -> receiver)
REVERSE_DIRECTION = 2  # dependent -> head
INTRA_DIRECTION = 3   # different subwords belonging to the same word


def validate_dependency_vocab(vocab):
    """Validate a JSON-serializable, contiguous training dependency vocabulary."""
    if not isinstance(vocab, dict) or not all(isinstance(k, str) and k for k in vocab):
        raise ValueError("dependency_vocab must be a nonempty string-to-integer dictionary")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in vocab.values()):
        raise ValueError("Dependency vocabulary IDs must be integers")
    if [vocab.get(k) for k in DEPENDENCY_SPECIAL_RELATIONS] != [0, 1, 2, 3]:
        raise ValueError("Dependency vocabulary requires PAD=0, UNK=1, SELF=2, INTRA=3")
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("Dependency vocabulary IDs must be unique and contiguous")
    return dict(vocab)


def _validate_record(record, location):
    words = record.get("token") if isinstance(record, dict) else None
    if not isinstance(words, list) or not words:
        raise ValueError("Expected nonempty dependency token list: " + location)
    for name in ("head", "deprel"):
        if not isinstance(record.get(name), list) or len(record[name]) != len(words):
            raise ValueError("Dependency %s/token lengths differ: %s" % (name, location))
    if any(not isinstance(label, str) or not label.strip() for label in record["deprel"]):
        raise ValueError("Dependency labels must be nonblank strings: " + location)
    if any(label in DEPENDENCY_SPECIAL_RELATIONS for label in record["deprel"]):
        raise ValueError("Raw parser labels collide with reserved dependency symbols: " + location)
    for head in record["head"]:
        if isinstance(head, bool) or not isinstance(head, int) or not 0 <= head <= len(words):
            raise ValueError("Expected 1-based dependency heads, root=0: " + location)


def build_dependency_vocab(train_file):
    """Read only the supplied training file; never inspect validation/test data."""
    records = _read_records(train_file)
    if not isinstance(records, list) or not records:
        raise ValueError("Training dependency records must be a nonempty list")
    labels = set()
    for index, record in enumerate(records):
        _validate_record(record, "%s record %d" % (train_file, index))
        labels.update(record["deprel"])
    vocab = {name: index for index, name in enumerate(DEPENDENCY_SPECIAL_RELATIONS)}
    vocab.update({name: index + len(vocab) for index, name in enumerate(sorted(labels))})
    return validate_dependency_vocab(vocab)


def _merge_key(record):
    # This is the frozen grouping key, not a new grouping policy. Additional
    # type attributes must agree before two existing records can be merged.
    return json.dumps([record["token"], record["head"], record["amr_tokens"],
                       sorted(record["amr_edges"], key=lambda edge: json.dumps(edge, sort_keys=True))],
                      ensure_ascii=False, sort_keys=True)


def _validate_merge_labels(records, path):
    seen = {}
    for index, record in enumerate(records):
        _validate_record(record, "%s record %d" % (path, index))
        # Let the reference loader retain its actionable error for these fields.
        if "amr_tokens" not in record or "amr_edges" not in record:
            continue
        key = _merge_key(record)
        if key in seen and seen[key] != record["deprel"]:
            raise ValueError("Conflicting dependency labels under the same frozen sentence merge key: %s record %d" % (path, index))
        seen[key] = record["deprel"]


def syntax_edge_attributes(record, example, vocab):
    """Unique source/destination edges covering exactly positive dep_adj.

    The dense GCN later adds its identity weight to the same diagonal positions;
    this must not be encoded as duplicated sparse edges. The original adjacency
    and lexical edge weights remain the source of edge magnitudes.
    """
    positions = {}
    for position, word in enumerate(example["word_ids"].tolist()):
        if word >= 0:
            positions.setdefault(word, []).append(position)
    edges = {}

    def add(source, target, relation, direction):
        key, attributes = (source, target), (relation, direction)
        if key in edges and edges[key] != attributes:
            raise ValueError("Conflicting parser attributes for one syntax edge: %s" % (key,))
        edges[key] = attributes

    for word, subwords in positions.items():
        for source in subwords:
            for target in subwords:
                same = source == target
                add(source, target, vocab["<self>" if same else "<intra_word>"],
                    SELF_DIRECTION if same else INTRA_DIRECTION)
        parent = record["head"][word] - 1
        if parent < 0 or parent == word:
            continue
        relation = vocab.get(record["deprel"][word], vocab["<unk>"])
        for source in positions.get(parent, ()):  # Twitter may omit an unencoded word.
            for target in subwords:
                add(source, target, relation, FORWARD_DIRECTION)
                add(target, source, relation, REVERSE_DIRECTION)
    # dep_adj is symmetric in v1; source/destination orientation nevertheless
    # remains explicit for typed message computation (receiver=destination).
    expected = {tuple(pair) for pair in example["dep_adj"].gt(0).nonzero(as_tuple=False).tolist()}
    if set(edges) != expected:
        raise ValueError("Typed syntax edges differ from frozen dependency topology")
    ordered = sorted(edges)
    values = torch.tensor([(s, t, *edges[(s, t)]) for s, t in ordered], dtype=torch.long)
    if not ordered:
        raise ValueError("Encoded sentence has no syntax self edges")
    return {"dep_src": values[:, 0], "dep_dst": values[:, 1],
            "dep_type": values[:, 2], "dep_direction": values[:, 3],
            "dep_edge_mask": torch.ones(len(ordered), dtype=torch.bool)}


class _TypedSyntaxMixin:
    def __init__(self, path, tokenizer, *, dependency_vocab=None, typed_enabled=True, **kwargs):
        if not isinstance(typed_enabled, bool):
            raise ValueError("typed_enabled must be boolean")
        self.typed_enabled = typed_enabled
        self.dependency_vocab = validate_dependency_vocab(dependency_vocab) if typed_enabled else None
        if typed_enabled:
            records = kwargs.pop("records", None)
            records = _read_records(path) if records is None else copy.deepcopy(records)
            if not isinstance(records, list):
                raise ValueError("Dependency records must be a list")
            _validate_merge_labels(records, str(path))
            kwargs["records"] = records
        super().__init__(path, tokenizer, **kwargs)
        self.stats["typed_syntax_enabled"] = typed_enabled
        if not typed_enabled:
            return
        self.stats["dependency_vocab_size"] = len(self.dependency_vocab)
        self.stats["dependency_vocab_source"] = "provided_training_vocabulary_never_expanded"
        self.stats["unknown_dependency_token_relations"] = sum(
            label not in self.dependency_vocab for record in self.records for label in record["deprel"])
        self.stats["expanded_dependency_edges"] = 0
        for record, example in zip(self.records, self.data):
            attributes = syntax_edge_attributes(record, example, self.dependency_vocab)
            example.update(attributes)
            self.stats["expanded_dependency_edges"] += int(attributes["dep_edge_mask"].numel())


class TypedSentenceDataset(_TypedSyntaxMixin, SentenceABSADataset):
    """Frozen sentence adapter plus sparse dependency attributes."""


class TypedTwitterDataset(_TypedSyntaxMixin, TwitterABSADataset):
    """Preserve Twitter's Unicode coverage policy; omit unencoded edge nodes."""


def typed_collate(examples):
    """Apply original collator, then independently pad sparse dependency edges."""
    if not examples:
        raise ValueError("Cannot collate an empty sentence batch")
    present = [DEP_EDGE_KEYS & set(example) for example in examples]
    if all(not keys for keys in present):
        return sentence_collate(examples)
    if any(keys != DEP_EDGE_KEYS for keys in present):
        raise ValueError("A typed syntax batch requires all edge fields on every example")
    result = sentence_collate([{k:v for k,v in example.items() if k not in DEP_EDGE_KEYS}
                               for example in examples])
    sizes = [example["dep_edge_mask"].numel() for example in examples]
    for key in DEP_EDGE_KEYS:
        output = examples[0][key].new_zeros((len(examples), max(sizes)))
        for index, (example, size) in enumerate(zip(examples, sizes)):
            if example[key].ndim != 1 or example[key].numel() != size:
                raise ValueError("Typed syntax edge fields must be aligned vectors")
            output[index, :size] = example[key]
        result[key] = output
    return result


__all__ = ["DEPENDENCY_SPECIAL_RELATIONS", "DEP_EDGE_KEYS", "build_dependency_vocab",
           "validate_dependency_vocab", "syntax_edge_attributes", "TypedSentenceDataset",
           "TypedTwitterDataset", "typed_collate"]
