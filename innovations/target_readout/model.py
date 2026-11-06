"""Explicit target readout, with identical parameters and evidence routing to v1.

The aspect mode duplicates the frozen forward to avoid stateful hooks or edits to
its source. Tests require every pre-classifier output to remain identical under
identical weights. The cls control delegates directly to the frozen model.
"""
import torch
from torch import nn
from torch.nn import functional as F

from ..v1_reference.model import MultiAspectDAGF, evidence_pool, routed_attention, sentence_cross_entropy


class TargetReadoutDAGF(MultiAspectDAGF):
    def __init__(self, *args, readout="aspect", **kwargs):
        if readout not in ("aspect", "cls"):
            raise ValueError("readout must be aspect or cls")
        super().__init__(*args, **kwargs)
        self.readout = readout

    def forward(self, batch, labels=None):
        if self.readout == "cls":
            return super().forward(batch, labels=labels)
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
        # Only this readout differs from the frozen forward: target query, not CLS.
        pooled = query
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
