"""Evidence ownership before graph fusion, with explicit shared/background routes.

The sentence encoder and the baseline's separate MLM branch each run once per
sentence batch. No gold polarity or opinion-span annotation is a model feature.
This is a latent evidence router, not supervised opinion-span extraction.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

# Prefer an installed project-level core, or use this checkout's saved payload.
# Neither layout is copied or patched.
from .baseline import load_baseline_module

CompleteDAGF = load_baseline_module("model").CompleteDAGF


def sentence_cross_entropy(logits, labels, aspect_valid):
    """Average aspects within each sentence, then average labelled sentences."""
    valid = aspect_valid.bool() & (labels != -100)
    if not valid.any():
        raise ValueError("Sentence CE requires at least one labelled aspect")
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                           ignore_index=-100, reduction="none").view_as(labels)
    weights = valid.to(loss.dtype)
    counts = weights.sum(-1)
    per_sentence = (loss * weights).sum(-1) / counts.clamp_min(1)
    return per_sentence[counts > 0].mean()


def evidence_pool(states, weights):
    """Multi-aspect pooling; an empty channel contributes exactly zero."""
    return torch.bmm(weights, states) / weights.sum(-1, keepdim=True).clamp_min(1e-8)


class OpinionRouter(nn.Module):
    """A+2 token channels: exclusive owners, a shared reservoir, background.

    Independent membership gates distribute the shared reservoir to ANY subset
    of aspects. R = exclusive + shared * membership is intentionally not a
    softmax over aspects; the same token may support several aspects fully.
    """

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.shared_head = nn.Linear(dim, 1)
        self.background_head = nn.Linear(dim, 1)
        self.membership_query = nn.Linear(dim, dim, bias=False)
        self.membership_key = nn.Linear(dim, dim, bias=False)
        self.membership_bias = nn.Parameter(torch.tensor(0.0))
        self.prior_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, tokens, queries, context_mask, aspect_valid, route_prior):
        valid = aspect_valid.bool()
        if not valid.any(dim=-1).all() or not context_mask.bool().any(dim=-1).all():
            raise ValueError("Each sentence needs a context token and a valid aspect")
        pair_mask = valid.unsqueeze(-1) & context_mask.bool().unsqueeze(1)
        scale = math.sqrt(tokens.size(-1))
        scores = torch.bmm(self.query(queries), self.key(tokens).transpose(1, 2)) / scale
        prior = F.softplus(self.prior_scale) * route_prior.to(tokens.dtype)
        scores = (scores + prior).masked_fill(~valid.unsqueeze(-1), torch.finfo(tokens.dtype).min)
        shared_score = self.shared_head(tokens).transpose(1, 2)
        # A single-aspect sentence has no multi-aspect shared reservoir.
        shared_score = shared_score.masked_fill(
            (valid.sum(-1) < 2).view(-1, 1, 1), torch.finfo(tokens.dtype).min)
        background_score = self.background_head(tokens).transpose(1, 2)
        channels = torch.softmax(torch.cat((scores, shared_score, background_score), dim=1), dim=1)
        channels = channels * context_mask.to(tokens.dtype).unsqueeze(1)
        exclusive = channels[:, :queries.size(1)] * pair_mask.to(tokens.dtype)
        shared = channels[:, -2]
        background = channels[:, -1]
        member_scores = torch.bmm(self.membership_query(queries),
                                  self.membership_key(tokens).transpose(1, 2)) / scale
        membership = torch.sigmoid(member_scores + prior + self.membership_bias)
        membership = membership * pair_mask.to(tokens.dtype)
        shared_routes = shared.unsqueeze(1) * membership
        routes = exclusive + shared_routes

        # Sentence-balanced unsupervised penalties, never opinion gold labels.
        token_count = context_mask.sum(-1).to(tokens.dtype).clamp_min(1)
        aspect_count = valid.sum(-1).to(tokens.dtype).clamp_min(1)
        mass = routes.sum(-1)
        density = (mass.sum(-1) / (token_count * aspect_count)).mean()
        distribution = routes / mass.unsqueeze(-1).clamp_min(1e-8)
        token_entropy = -(distribution * distribution.clamp_min(1e-8).log()).sum(-1)
        token_entropy = token_entropy / token_count.clamp_min(2).log().unsqueeze(-1)
        route_entropy = ((token_entropy * valid).sum(-1) / aspect_count).mean()
        # Normalized readouts are otherwise invariant to shrinking every route
        # toward background. A one-token mass floor prevents that trivial
        # solution; token entropy favors concentrated evidence above the floor.
        coverage = ((F.relu(1.0 - mass).square() * valid).sum(-1) / aspect_count).mean()
        sparsity = density + 0.1 * route_entropy + coverage
        exclusive_mass = exclusive.sum(1)
        owners = exclusive / exclusive_mass.unsqueeze(1).clamp_min(1e-8)
        entropy = -(owners * owners.clamp_min(1e-8).log()).sum(1)
        # Entropy applies only to the exclusive reservoir, not shared evidence.
        normalizer = aspect_count.clamp_min(2).log()
        exclusivity = ((entropy * exclusive_mass).sum(-1) / token_count / normalizer).mean()
        members = membership.sum(1)
        # Sharing has a small cost; a shared token should reach >=2 aspects.
        underused = F.relu(2.0 - members).square() / 4.0
        sharing = ((shared * (underused + 0.1)).sum(-1) / token_count).mean()
        return {"R": routes, "R_exclusive": exclusive, "R_shared": shared_routes,
                "background": background, "shared_channel": shared,
                "membership": membership, "sparsity_loss": sparsity,
                "coverage_loss": coverage, "route_entropy_loss": route_entropy,
                "exclusivity_loss": exclusivity, "sharing_loss": sharing}


def routed_attention(module, sequence, queries, context_mask, aspect_mask, relation_bias, routes):
    """Reuse the baseline attention parameters for A queries without B*A graphs."""
    batch, length, dim = sequence.shape
    aspects = queries.size(1)
    q = module.query(queries).view(batch, aspects, module.heads, module.head_dim).transpose(1, 2)
    k = module.key(sequence).view(batch, length, module.heads, module.head_dim).transpose(1, 2)
    v = module.value(sequence).view(batch, length, module.heads, module.head_dim).transpose(1, 2)
    aspect_weights = aspect_mask.to(sequence.dtype)
    aspect_weights = aspect_weights / aspect_weights.sum(-1, keepdim=True).clamp_min(1)
    prior = torch.bmm(aspect_weights, relation_bias.to(sequence.dtype))
    scores = (torch.matmul(q, k.transpose(-1, -2)) + prior.unsqueeze(1)) / math.sqrt(module.head_dim)
    scores = scores.masked_fill(~context_mask.bool()[:, None, None, :], torch.finfo(scores.dtype).min)
    # Multiplicative routing preserves a truly empty route as zero evidence.
    weights = torch.softmax(scores, dim=-1) * routes.unsqueeze(1)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    attended = torch.matmul(module.dropout(weights), v).transpose(1, 2).reshape(batch, aspects, dim)
    return module.output(attended)


class SharedEvidenceInteraction(nn.Module):
    """Send shared evidence only; private features never become other messages.

    Relation IDs encode coordination, contrast and comparison, not sentiment
    signs. Shared overlap gates every message, including structurally related
    aspects, so inter-aspect attention cannot bypass evidence ownership.
    """

    def __init__(self, dim, dropout):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.relation_bias = nn.Embedding(4, 1, padding_idx=0)
        self.gate = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, fused, shared_features, shared_routes, valid, relations):
        size = valid.size(1)
        pair_mask = valid.unsqueeze(1) & valid.unsqueeze(2)
        pair_mask = pair_mask & ~torch.eye(size, dtype=torch.bool, device=valid.device).unsqueeze(0)
        norm = shared_routes.square().sum(-1, keepdim=True).clamp_min(1e-16).sqrt()
        unit = shared_routes / norm
        overlap = torch.bmm(unit, unit.transpose(1, 2)).clamp(0, 1) * pair_mask
        scores = torch.bmm(self.query(fused), self.key(shared_features).transpose(1, 2))
        scores = scores / math.sqrt(fused.size(-1)) + self.relation_bias(relations).squeeze(-1)
        scores = scores.masked_fill(~pair_mask, torch.finfo(scores.dtype).min)
        # Do not renormalize after overlap: arbitrarily tiny overlap must not
        # create a full-sized cross-aspect message.
        weights = torch.softmax(scores, dim=-1) * overlap * pair_mask
        message = torch.bmm(weights, self.value(shared_features))
        gate = torch.sigmoid(self.gate(torch.cat((fused, message), dim=-1)))
        result = self.norm(fused + self.dropout(gate * message))
        return result * valid.unsqueeze(-1), weights, overlap


class MultiAspectDAGF(CompleteDAGF):
    """Additive DAGF variant with sentence encoding and latent evidence routing."""

    def __init__(self, *args, use_routing=True, use_inter_aspect=True,
                 use_structure_prior=True, sparsity_weight=0.01,
                 exclusivity_weight=0.01, sharing_weight=0.01,
                 baseline_regularization_weight=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        for value in (sparsity_weight, exclusivity_weight, sharing_weight, baseline_regularization_weight):
            if value < 0:
                raise ValueError("Loss weights must be nonnegative")
        self.use_routing = use_routing
        self.use_inter_aspect = use_inter_aspect
        self.use_structure_prior = use_structure_prior
        self.sparsity_weight = sparsity_weight
        self.exclusivity_weight = exclusivity_weight
        self.sharing_weight = sharing_weight
        self.baseline_regularization_weight = baseline_regularization_weight
        self.router = OpinionRouter(self.hidden_dim)
        self.routing_norm = nn.LayerNorm(self.hidden_dim)
        self.private_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.shared_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.evidence_norm = nn.LayerNorm(self.hidden_dim)
        self.interaction = SharedEvidenceInteraction(self.hidden_dim, self.dropout.p)
        for module in (self.router, self.routing_norm, self.private_projection,
                       self.shared_projection, self.evidence_norm, self.interaction):
            module.apply(self._initialize_added_module)

    def forward(self, batch, labels=None):
        context = batch["context_mask"].bool()
        masks = batch["aspect_mask"].bool()
        valid = batch["aspect_valid"].bool()
        text_output = self.encoder(input_ids=batch["text_ids"],
                                   attention_mask=batch["text_attention_mask"],
                                   return_dict=True).last_hidden_state
        text_states = self.text_projection(text_output)
        query = self.aspect_projection(evidence_pool(text_output, masks.to(text_output.dtype)))
        query = query * valid.unsqueeze(-1)
        prompt_logits = self.mlm(input_ids=batch["prompt_ids"],
                                 attention_mask=batch["prompt_attention_mask"],
                                 return_dict=True).logits
        prompt_states = self.prompt_projection(prompt_logits)
        syntax_input = self.dropout(self.syntax_input_norm(text_states + prompt_states))
        syntax_input = syntax_input * context.unsqueeze(-1)
        relation_bias = batch["relation_bias"].to(text_states.dtype)
        graph = batch["dep_adj"].to(text_states.dtype) + relation_bias.clamp_min(0)
        syntax_states = syntax_input
        for layer in self.syntax_layers:
            syntax_states = layer(syntax_states, graph, context)
        cnn = F.gelu(self.syntax_cnn(syntax_states.transpose(1, 2)))
        cnn = self.cnn_projection(F.max_pool1d(cnn, 3, 1, 1).transpose(1, 2))
        lengths = batch["text_attention_mask"].sum(1).long().clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(syntax_input, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.text_lstm(packed)
        lstm, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True,
                                                  total_length=syntax_input.size(1))
        lstm = self.lstm_projection(lstm)
        semantic_states = self.dropout(text_states) * context.unsqueeze(-1)
        if self.use_amr:
            relations = self.amr_relations(batch["amr_rel"])
            for layer in self.semantic_layers:
                semantic_states = layer(semantic_states, batch["amr_src"], batch["amr_dst"],
                                        relations, batch["amr_edge_mask"], context)

        # Both graph streams are still separate at this point. Ownership is
        # established before graph fusion and before any aspect interaction.
        tokens = self.routing_norm(text_states + syntax_states + semantic_states)
        prior = batch["route_prior"] if self.use_structure_prior else torch.zeros_like(batch["route_prior"])
        if self.use_routing:
            routing = self.router(tokens, query, context, valid, prior)
        else:
            # Same sentence encoder, queries, graph/fusion/classifier budget;
            # remove learned ownership, all tokens are available to each aspect.
            all_context = (valid.unsqueeze(-1) & context.unsqueeze(1)).to(tokens.dtype)
            routing = {"R": all_context, "R_exclusive": all_context,
                       "R_shared": all_context, "background": tokens.new_zeros(context.shape),
                       "shared_channel": context.to(tokens.dtype), "membership": all_context,
                       **{name: tokens.sum() * 0 for name in
                          ("sparsity_loss", "exclusivity_loss", "sharing_loss",
                           "coverage_loss", "route_entropy_loss")}}
        routes = routing["R"]
        cnn_summary = routed_attention(self.cnn_attention, cnn, query, context, masks, relation_bias, routes)
        lstm_summary = routed_attention(self.lstm_attention, lstm, query, context, masks, relation_bias, routes)
        syn = self.syntax_readout(torch.cat((cnn_summary, lstm_summary), dim=-1))
        sem = self.semantic_norm(evidence_pool(semantic_states, routes))
        # Only valid aspects enter AGF: padding never affects branch losses.
        fusion = self.fusion(syn[valid], sem[valid], labels=(labels[valid] if
                             labels is not None and self.use_agf and self.baseline_regularization_weight > 0 else None))
        flat_fused = fusion["fused"] if self.use_agf else self.plain_fusion(torch.cat((syn[valid], sem[valid]), -1))
        fused = tokens.new_zeros((*valid.shape, self.hidden_dim))
        fused[valid] = flat_fused
        exclusive = evidence_pool(tokens, routing["R_exclusive"])
        shared = evidence_pool(tokens, routing["R_shared"])
        private_mass = routing["R_exclusive"].sum(-1, keepdim=True)
        shared_mass = routing["R_shared"].sum(-1, keepdim=True)
        total_mass = (private_mass + shared_mass).clamp_min(1e-8)
        private_features = self.private_projection(exclusive) * private_mass / total_mass
        shared_features = self.shared_projection(shared) * shared_mass / total_mass
        fused = self.evidence_norm(fused + private_features + shared_features) * valid.unsqueeze(-1)
        interaction_weights = tokens.new_zeros((*valid.shape, valid.size(1)))
        overlap = interaction_weights
        if self.use_inter_aspect:
            relations = batch["aspect_relation"] if self.use_structure_prior else torch.zeros_like(batch["aspect_relation"])
            fused, interaction_weights, overlap = self.interaction(fused, shared_features,
                                                                   routing["R_shared"], valid, relations)
        pooled = text_states[:, :1].expand(-1, valid.size(1), -1)
        logits = self.classifier(self.dropout(self.output_norm(torch.cat((fused, pooled), dim=-1))))
        logits = logits * valid.unsqueeze(-1)
        kl = F.kl_div(F.log_softmax(fusion["syn_logits"], -1),
                      F.softmax(fusion["sem_logits"], -1), reduction="batchmean")
        baseline_loss = self.kl_weight * kl + (fusion["auxiliary_loss"] if self.use_agf else kl * 0)
        regularization = (self.sparsity_weight * routing["sparsity_loss"]
                          + self.exclusivity_weight * routing["exclusivity_loss"]
                          + self.sharing_weight * routing["sharing_loss"]
                          + self.baseline_regularization_weight * baseline_loss)
        return {**routing, "logits": logits, "regularization_loss": regularization,
                "kl_loss": kl, "auxiliary_loss": fusion["auxiliary_loss"],
                "aspect_attention": interaction_weights, "shared_overlap": overlap,
                "private_features": private_features, "shared_features": shared_features}
