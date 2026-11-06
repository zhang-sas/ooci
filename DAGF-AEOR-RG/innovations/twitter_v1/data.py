"""Twitter-only compatibility for tokenizers that omit decorative Unicode words.

No words are replaced, no data files or aspect spans/labels are rewritten.
Unencoded graph nodes are omitted; an entirely unencoded aspect remains an error.
When every word is encoded, all v1 tensors stay bitwise identical.
"""
import torch
from ..v1_reference.data import (
    SentenceABSADataset as FrozenSentenceDataset, POLARITY_IDS,
    DATASET_FOLDERS, build_amr_vocab, sentence_collate,
)


class TwitterABSADataset(FrozenSentenceDataset):
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
        missing = sorted(set(range(len(words))) - set(word_to_positions))
        target_words = {i for aspect in aspects for i in range(aspect["from"], aspect["to"])}
        if not aspect_mask.any(dim=1).all():
            raise ValueError("Tokenizer lost an entire ASPECT: %s sentence %d; repair tokenizer assets, do not drop labels" % (self.path, sentence_index))
        # Preserve exactly the tokenizer-produced IDs. Unencoded non-aspect
        # symbols get no fake embedding/node; their original word indices stay.
        for index in missing:
            word_to_positions[index] = []
        nonaspect = [i for i in missing if i not in target_words]
        partial = [ai for ai, a in enumerate(aspects) if any(i in missing for i in range(a["from"], a["to"]))]
        metadata["unencoded_word_indices"] = missing
        metadata["unencoded_nonaspect_word_indices"] = nonaspect
        metadata["partially_unencoded_aspect_indices"] = partial
        for name in ("sentences_with_unencoded_words", "unencoded_words", "unencoded_nonaspect_words", "partially_unencoded_aspects"):
            self.stats.setdefault(name, 0)
        self.stats["sentences_with_unencoded_words"] += int(bool(missing))
        self.stats["unencoded_words"] += len(missing)
        self.stats["unencoded_nonaspect_words"] += len(nonaspect)
        self.stats["partially_unencoded_aspects"] += len(partial)
        self.stats["tokenizer_coverage_policy"] = "preserve encoded IDs and all labels/spans; omit unencoded graph positions; require nonempty evidence span for each aspect"
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



SentenceABSADataset = TwitterABSADataset
