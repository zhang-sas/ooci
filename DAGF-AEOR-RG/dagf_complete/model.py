"""Default completed DAGF model; this module never imports the legacy model.

This is an explicit engineering interpretation of the paper, not recovered
author weights. AMR relation embeddings are learned from the training relation
vocabulary. The APARN path-embedding lookup is intentionally not guessed.
The CNN preserves token positions so relation-aware attention and all graph
indices share one subword coordinate system. The LSTM is unidirectional with
an independent 300-dimensional hidden state by default.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import RobertaForMaskedLM, RobertaModel

from .fusion import AuxiliaryGraphFusion


def masked_mean(states, mask):
    weights = mask.to(states.dtype).unsqueeze(-1)
    return (states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class DenseSyntaxGCN(nn.Module):
    """Registered, residual GCN on a masked knowledge-enhanced syntax graph."""

    def __init__(self, dim, dropout):
        super().__init__()
        self.message = nn.Linear(dim, dim, bias=False)
        self.self_projection = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states, adjacency, node_mask):
        mask = node_mask.to(states.dtype)
        valid_pairs = mask.unsqueeze(2) * mask.unsqueeze(1)
        graph = adjacency.to(states.dtype).clamp_min(0.0) * valid_pairs
        identity = torch.eye(states.size(1), device=states.device, dtype=states.dtype)
        graph = graph + identity.unsqueeze(0) * valid_pairs
        graph = graph / graph.sum(dim=-1, keepdim=True).clamp_min(1.0)
        update = torch.bmm(graph, self.message(states)) + self.self_projection(states)
        result = self.norm(states + self.dropout(F.gelu(update)))
        return result * mask.unsqueeze(-1)


class SparseAMRGCN(nn.Module):
    """Relation-aware AMR messages without a dense [B,L,L,1024] embedding."""

    def __init__(self, dim, relation_dim, dropout):
        super().__init__()
        self.source_projection = nn.Linear(dim, dim, bias=False)
        self.relation_projection = nn.Linear(relation_dim, dim, bias=False)
        self.relation_gate = nn.Linear(relation_dim, 1)
        self.self_projection = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states, source, target, relations, edge_mask, node_mask):
        batch_size, length, dim = states.shape
        source_valid = (source >= 0) & (source < length)
        target_valid = (target >= 0) & (target < length)
        source = source.clamp(min=0, max=length - 1)
        target = target.clamp(min=0, max=length - 1)
        source_nodes = torch.gather(node_mask, 1, source).bool()
        target_nodes = torch.gather(node_mask, 1, target).bool()
        active = edge_mask.bool() & source_valid & target_valid & source_nodes & target_nodes

        source_states = torch.gather(states, 1, source.unsqueeze(-1).expand(-1, -1, dim))
        gate = torch.sigmoid(self.relation_gate(relations)) * active.unsqueeze(-1).to(states.dtype)
        messages = self.source_projection(source_states) + self.relation_projection(relations)
        messages = self.dropout(F.gelu(messages)) * gate

        aggregated = states.new_zeros((batch_size, length, dim))
        aggregated.scatter_add_(1, target.unsqueeze(-1).expand(-1, -1, dim), messages)
        degree = states.new_zeros((batch_size, length, 1))
        degree.scatter_add_(1, target.unsqueeze(-1), gate)
        aggregated = aggregated / degree.clamp_min(1.0)
        update = self.self_projection(states) + aggregated
        result = self.norm(states + self.dropout(F.gelu(update)))
        return result * node_mask.to(states.dtype).unsqueeze(-1)


class RelationAwareAspectAttention(nn.Module):
    """Aspect query attends to token features with an explicit relation prior."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        if dim % heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, aspect, context_mask, aspect_mask, relation_bias):
        batch_size, length, dim = sequence.shape
        query = self.query(aspect).view(batch_size, self.heads, 1, self.head_dim)
        key = self.key(sequence).view(batch_size, length, self.heads, self.head_dim).transpose(1, 2)
        value = self.value(sequence).view(batch_size, length, self.heads, self.head_dim).transpose(1, 2)

        aspect_weights = aspect_mask.to(sequence.dtype)
        aspect_weights = aspect_weights / aspect_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        prior = torch.bmm(aspect_weights.unsqueeze(1), relation_bias.to(sequence.dtype))
        scores = (torch.matmul(query, key.transpose(-1, -2)) + prior.unsqueeze(1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~context_mask.bool().unsqueeze(1).unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(weights, value).reshape(batch_size, dim)
        return self.output(attended)


class CompleteDAGF(nn.Module):
    """Completed syntax/AMR/AGF experiment with a label-free inference path.

    Labels only supervise the auxiliary heads and learned reliability estimates.
    They do not enter the classifier feature calculation, including during
    training. ``use_amr=False`` bypasses AMR propagation; ``use_agf=False`` uses
    ordinary projected feature concatenation. These are explicit ablations.
    """

    def __init__(
        self,
        roberta_path="./roberta",
        hidden_dim=768,
        lstm_hidden=300,
        num_layers=1,
        attention_heads=8,
        dropout=0.3,
        alpha=0.8,
        beta=0.6,
        kl_weight=1.5,
        num_amr_relations=3,
        aux_loss_weight=0.1,
        gradient_detach=True,
        gradient_checkpointing=False,
        amr_dim=128,
        use_amr=True,
        use_agf=True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        if num_amr_relations < 3:
            raise ValueError("AMR vocabulary must include pad, unknown and self relations")
        if hidden_dim < 1 or lstm_hidden < 1 or amr_dim < 1:
            raise ValueError("All hidden dimensions must be positive")

        self.hidden_dim = hidden_dim
        self.lstm_hidden = lstm_hidden
        self.kl_weight = kl_weight
        self.use_amr = use_amr
        self.use_agf = use_agf
        self.mlm = RobertaForMaskedLM.from_pretrained(roberta_path)
        self.encoder = RobertaModel.from_pretrained(roberta_path, add_pooling_layer=False)
        self.pad_token_id = self.encoder.config.pad_token_id
        self.bos_token_id = self.encoder.config.bos_token_id
        self.eos_token_id = self.encoder.config.eos_token_id
        encoder_dim = self.encoder.config.hidden_size
        if self.mlm.config.vocab_size != self.encoder.config.vocab_size:
            raise ValueError("The two RoBERTa encoders must have the same vocabulary")
        if gradient_checkpointing:
            self.mlm.gradient_checkpointing_enable()
            self.encoder.gradient_checkpointing_enable()
            self.mlm.config.use_cache = False
            self.encoder.config.use_cache = False

        self.text_projection = nn.Linear(encoder_dim, hidden_dim)
        self.aspect_projection = nn.Linear(encoder_dim, hidden_dim)
        self.prompt_projection = nn.Linear(self.mlm.config.vocab_size, hidden_dim)
        self.syntax_input_norm = nn.LayerNorm(hidden_dim)
        self.syntax_layers = nn.ModuleList([
            DenseSyntaxGCN(hidden_dim, dropout) for _ in range(num_layers)
        ])
        # Ten CNN channels retain the paper's channel count. Stride one and
        # symmetric padding preserve subword positions for relation alignment.
        self.syntax_cnn = nn.Conv1d(hidden_dim, 10, kernel_size=5, padding=2)
        self.cnn_projection = nn.Linear(10, hidden_dim)
        self.text_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.lstm_projection = nn.Linear(lstm_hidden, hidden_dim)
        self.cnn_attention = RelationAwareAspectAttention(hidden_dim, attention_heads, dropout)
        self.lstm_attention = RelationAwareAspectAttention(hidden_dim, attention_heads, dropout)
        self.syntax_readout = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )

        self.amr_relations = nn.Embedding(num_amr_relations, amr_dim, padding_idx=0)
        self.semantic_layers = nn.ModuleList([
            SparseAMRGCN(hidden_dim, amr_dim, dropout) for _ in range(num_layers)
        ])
        self.semantic_norm = nn.LayerNorm(hidden_dim)
        self.fusion = AuxiliaryGraphFusion(
            dim=hidden_dim,
            num_classes=3,
            alpha=alpha,
            beta=beta,
            dropout=dropout,
            aux_loss_weight=aux_loss_weight,
            gradient_detach=gradient_detach,
        )
        self.plain_fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
        )
        self.output_norm = nn.LayerNorm(2 * hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_dim, 3)

        # Initialize only the added modules; pretrained encoders remain intact.
        for name, module in self.named_children():
            if name not in ("mlm", "encoder"):
                module.apply(self._initialize_added_module)

    @staticmethod
    def _initialize_added_module(module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LSTM):
            for name, parameter in module.named_parameters(recurse=False):
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(parameter)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)

    def forward(self, batch, labels=None):
        context_mask = batch["context_mask"].bool()
        aspect_mask = batch["aspect_mask"].bool()
        text_output = self.encoder(
            input_ids=batch["text_ids"],
            attention_mask=batch["text_attention_mask"],
            return_dict=True,
        ).last_hidden_state
        text_states = self.text_projection(text_output)
        pooled = text_states[:, 0]

        aspect_output = self.encoder(
            input_ids=batch["aspect_ids"],
            attention_mask=batch["aspect_attention_mask"],
            return_dict=True,
        ).last_hidden_state
        aspect_tokens = batch["aspect_attention_mask"].bool()
        for special_id in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
            if special_id is not None:
                aspect_tokens = aspect_tokens & (batch["aspect_ids"] != special_id)
        aspect = self.aspect_projection(masked_mean(aspect_output, aspect_tokens))

        prompt_logits = self.mlm(
            input_ids=batch["prompt_ids"],
            attention_mask=batch["prompt_attention_mask"],
            return_dict=True,
        ).logits
        prompt_states = self.prompt_projection(prompt_logits)
        if prompt_states.size(1) != text_states.size(1):
            raise ValueError("Prompt and text input must use the same padded sequence length")
        syntax_input = self.syntax_input_norm(text_states + prompt_states)
        syntax_input = self.dropout(syntax_input) * context_mask.to(text_states.dtype).unsqueeze(-1)

        relation_bias = batch["relation_bias"].to(text_states.dtype)
        syntax_graph = batch["dep_adj"].to(text_states.dtype) + relation_bias.clamp_min(0.0)
        syntax_states = syntax_input
        for graph_layer in self.syntax_layers:
            syntax_states = graph_layer(syntax_states, syntax_graph, context_mask)
        cnn = F.gelu(self.syntax_cnn(syntax_states.transpose(1, 2)))
        cnn = F.max_pool1d(cnn, kernel_size=3, stride=1, padding=1)
        cnn = self.cnn_projection(cnn.transpose(1, 2))

        lengths = batch["text_attention_mask"].sum(dim=1).long().clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            syntax_input, lengths, batch_first=True, enforce_sorted=False,
        )
        packed_output, _ = self.text_lstm(packed)
        lstm, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=syntax_input.size(1),
        )
        lstm = self.lstm_projection(lstm)
        cnn_summary = self.cnn_attention(cnn, aspect, context_mask, aspect_mask, relation_bias)
        lstm_summary = self.lstm_attention(lstm, aspect, context_mask, aspect_mask, relation_bias)
        syn = self.syntax_readout(torch.cat((cnn_summary, lstm_summary), dim=-1))

        semantic_states = self.dropout(text_states) * context_mask.to(text_states.dtype).unsqueeze(-1)
        if self.use_amr:
            relations = self.amr_relations(batch["amr_rel"])
            for graph_layer in self.semantic_layers:
                semantic_states = graph_layer(
                    semantic_states,
                    batch["amr_src"],
                    batch["amr_dst"],
                    relations,
                    batch["amr_edge_mask"],
                    context_mask,
                )
        sem = self.semantic_norm(masked_mean(semantic_states, aspect_mask))

        fusion_output = self.fusion(syn, sem, labels=labels if self.use_agf else None)
        if self.use_agf:
            fused = fusion_output["fused"]
            auxiliary_loss = fusion_output["auxiliary_loss"]
        else:
            fused = self.plain_fusion(torch.cat((syn, sem), dim=-1))
            auxiliary_loss = syn.new_zeros(())

        # Direction is explicit: KL(P_sem || P_syn), averaged over examples.
        kl_loss = F.kl_div(
            F.log_softmax(fusion_output["syn_logits"], dim=-1),
            F.softmax(fusion_output["sem_logits"], dim=-1),
            reduction="batchmean",
        )
        classifier_input = self.output_norm(torch.cat((fused, pooled), dim=-1))
        logits = self.classifier(self.dropout(classifier_input))
        return {
            "logits": logits,
            "regularization_loss": self.kl_weight * kl_loss + auxiliary_loss,
            "kl_loss": kl_loss,
            "auxiliary_loss": auxiliary_loss,
            "syn_logits": fusion_output["syn_logits"],
            "sem_logits": fusion_output["sem_logits"],
            "gate": fusion_output["gate"],
        }
