"""Factorial routing/RG controls and an actual shared-channel ablation."""
import math
import torch
from torch.nn import functional as F

from ..v1_reference.model import OpinionRouter
from ..v1_relation_gate.model import V1RelationGateDAGF


class ExclusiveOnlyRouter(OpinionRouter):
    """Remove shared-channel probability before normalization, retaining background.

    Original parameter names/initial values are retained for a matched backbone.
    Shared-head/membership parameters are inactive and have no gradients. This
    differs from setting sharing_weight=0, which leaves sharing available.
    """

    def forward(self, tokens, queries, context_mask, aspect_valid, route_prior):
        valid = aspect_valid.bool()
        if not valid.any(dim=-1).all() or not context_mask.bool().any(dim=-1).all():
            raise ValueError("Each sentence needs context and a valid aspect")
        pair_mask = valid.unsqueeze(-1) & context_mask.bool().unsqueeze(1)
        scores = torch.bmm(self.query(queries), self.key(tokens).transpose(1, 2)) / math.sqrt(tokens.size(-1))
        prior = F.softplus(self.prior_scale) * route_prior.to(tokens.dtype)
        scores = (scores + prior).masked_fill(~valid.unsqueeze(-1), torch.finfo(tokens.dtype).min)
        background_score = self.background_head(tokens).transpose(1, 2)
        channels = torch.softmax(torch.cat((scores, background_score), dim=1), dim=1)
        channels = channels * context_mask.to(tokens.dtype).unsqueeze(1)
        exclusive = channels[:, :queries.size(1)] * pair_mask.to(tokens.dtype)
        background = channels[:, -1]
        shared = torch.zeros_like(background)
        membership = torch.zeros_like(exclusive)
        token_count = context_mask.sum(-1).to(tokens.dtype).clamp_min(1)
        aspect_count = valid.sum(-1).to(tokens.dtype).clamp_min(1)
        mass = exclusive.sum(-1)
        density = (mass.sum(-1) / (token_count * aspect_count)).mean()
        distribution = exclusive / mass.unsqueeze(-1).clamp_min(1e-8)
        entropy = -(distribution * distribution.clamp_min(1e-8).log()).sum(-1)
        entropy = entropy / token_count.clamp_min(2).log().unsqueeze(-1)
        route_entropy = ((entropy * valid).sum(-1) / aspect_count).mean()
        coverage = ((F.relu(1.0 - mass).square() * valid).sum(-1) / aspect_count).mean()
        sparsity = density + 0.1 * route_entropy + coverage
        exclusive_mass = exclusive.sum(1)
        owners = exclusive / exclusive_mass.unsqueeze(1).clamp_min(1e-8)
        owner_entropy = -(owners * owners.clamp_min(1e-8).log()).sum(1)
        exclusivity = ((owner_entropy * exclusive_mass).sum(-1) / token_count / aspect_count.clamp_min(2).log()).mean()
        return dict(R=exclusive, R_exclusive=exclusive, R_shared=membership,
                    background=background, shared_channel=shared, membership=membership,
                    sparsity_loss=sparsity, coverage_loss=coverage, route_entropy_loss=route_entropy,
                    exclusivity_loss=exclusivity, sharing_loss=tokens.sum() * 0)


class PaperAblationDAGF(V1RelationGateDAGF):
    def __init__(self, *args, use_shared_channel=True, **kwargs):
        if not isinstance(use_shared_channel, bool):
            raise ValueError("use_shared_channel must be bool")
        if not use_shared_channel and kwargs.get("use_routing", True) is False:
            raise ValueError("Shared-channel ablation requires learned routing")
        super().__init__(*args, **kwargs)
        self.use_shared_channel = use_shared_channel
        if not use_shared_channel:
            # Preserve every existing module's initialization and the global RNG.
            with torch.random.fork_rng(devices=[]):
                router = ExclusiveOnlyRouter(self.hidden_dim)
            router.load_state_dict(self.router.state_dict(), strict=True)
            router.to(device=self.router.query.weight.device, dtype=self.router.query.weight.dtype)
            router.train(self.router.training)
            self.router = router

