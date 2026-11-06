"""Auxiliary graph fusion for the default completed DAGF implementation.

This is an explicit engineering interpretation of paper Section 3.4, not an
author-provided implementation. The paper uses true-label prediction errors in
its gate, but does not specify how that gate works for unlabelled inference.
Here reliability regressors learn those errors during training; their predictions
drive the gate in both training and inference. The final prediction therefore
never requires, or directly consumes, a gold label.

The paper describes a gate in (0, 1) while naming ReLU, so this implementation
uses sigmoid. Equation (20) squares the semantic error gradient; that choice is
used rather than the product of gradients in the original repository fragment.
For the label-free gradient feature, both channels are compared with a detached
soft consensus target. The gradient is calculated analytically, which also works
inside torch.no_grad() and avoids second-order autograd graphs by default.

All parameters are registered in __init__. This module does not alter the
original model, data loader, checkpoints, or training entry point.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class AuxiliaryGraphFusion(nn.Module):
    """Fuse two [batch, dim] features without consulting inference labels.

    Args:
        dim: Common syntactic/semantic feature dimension.
        num_classes: Number of sentiment classes.
        alpha: Contribution of the projected syntactic representation.
        beta: Contribution of the projected semantic representation.
        dropout: Dropout used by the gate and reliability regressors.
        aux_loss_weight: Scale of branch CE and reliability supervision.
        gradient_detach: Detach analytic gradients before building the auxiliary
            feature. False keeps their ordinary differentiable tensor expression;
            it still does not invoke torch.autograd.grad.
    """

    def __init__(
        self,
        dim=768,
        num_classes=3,
        alpha=0.8,
        beta=0.6,
        dropout=0.3,
        aux_loss_weight=0.1,
        gradient_detach=True,
    ):
        super().__init__()
        if dim < 1 or num_classes < 2:
            raise ValueError("AGF requires dim >= 1 and num_classes >= 2")
        self.dim = int(dim)
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.aux_loss_weight = float(aux_loss_weight)
        self.gradient_detach = bool(gradient_detach)
        self.eps = 1e-6

        self.syn_norm = nn.LayerNorm(self.dim)
        self.sem_norm = nn.LayerNorm(self.dim)
        self.syn_head = nn.Linear(self.dim, self.num_classes)
        self.sem_head = nn.Linear(self.dim, self.num_classes)
        self.syn_projection = nn.Linear(self.dim, self.dim)
        self.sem_projection = nn.Linear(self.dim, self.dim)

        reliability_hidden = max(32, self.dim // 4)
        self.syn_reliability = self._make_reliability_head(
            self.dim + self.num_classes, reliability_hidden, dropout
        )
        self.sem_reliability = self._make_reliability_head(
            self.dim + self.num_classes, reliability_hidden, dropout
        )
        self.gate_network = nn.Sequential(
            nn.Linear(2 * self.dim + 2, self.dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.dim, self.dim),
            nn.Sigmoid(),
        )
        # No bias: exactly zero auxiliary gradients should add no new feature.
        self.auxiliary_projection = nn.Linear(self.dim, self.dim, bias=False)
        self.output_norm = nn.LayerNorm(self.dim)

    @staticmethod
    def _make_reliability_head(input_dim, hidden_dim, dropout):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def _normalize_error_targets(self, errors):
        """Paper min-max scaling, with a non-degenerate singleton fallback."""
        smallest = errors.amin(dim=0, keepdim=True)
        largest = errors.amax(dim=0, keepdim=True)
        spread = largest - smallest
        scaled = (errors - smallest) / spread.clamp_min(self.eps)
        # L2 distance between a probability vector and a one-hot label is <= sqrt2.
        bounded = (errors / math.sqrt(2.0)).clamp(0.0, 1.0)
        return torch.where(spread > self.eps, scaled, bounded)

    def _analytic_error_gradient(self, probabilities, target, head):
        """d ||softmax(H W^T + b) - target||_2 / dH, for fixed target."""
        delta = probabilities - target
        length = delta.square().sum(dim=-1, keepdim=True).clamp_min(
            self.eps * self.eps
        ).sqrt()
        probability_gradient = delta / length
        logit_gradient = probabilities * (
            probability_gradient
            - (probability_gradient * probabilities).sum(dim=-1, keepdim=True)
        )
        return logit_gradient.matmul(head.weight)

    def _label_targets(self, labels, device):
        if labels.ndim == 1:
            class_ids = labels.to(device=device, dtype=torch.long)
            targets = F.one_hot(class_ids, num_classes=self.num_classes)
        elif labels.ndim == 2 and labels.size(-1) == self.num_classes:
            class_ids = labels.to(device=device).argmax(dim=-1)
            targets = F.one_hot(class_ids, num_classes=self.num_classes)
        else:
            raise ValueError("AGF labels must have shape [batch] or [batch, classes]")
        return class_ids, targets

    def forward(self, syn, sem, labels=None):
        if syn.ndim != 2 or sem.shape != syn.shape or syn.size(-1) != self.dim:
            raise ValueError("AGF expects matching [batch, dim] branch features")

        syn_features = self.syn_norm(syn)
        sem_features = self.sem_norm(sem)
        syn_logits = self.syn_head(syn_features)
        sem_logits = self.sem_head(sem_features)
        syn_probabilities = F.softmax(syn_logits, dim=-1)
        sem_probabilities = F.softmax(sem_logits, dim=-1)

        # These are predicted normalized errors, never true-label errors. Keeping
        # them unscaled across a batch also makes singleton inference well-defined.
        syn_error = self.syn_reliability(
            torch.cat((syn_features, syn_probabilities.detach()), dim=-1)
        )
        sem_error = self.sem_reliability(
            torch.cat((sem_features, sem_probabilities.detach()), dim=-1)
        )
        gate = self.gate_network(
            torch.cat((syn_error, sem_error, syn_features, sem_features), dim=-1)
        )

        consensus = 0.5 * (
            syn_probabilities.detach() + sem_probabilities.detach()
        )
        # Normalized H_sem is the feature representation differentiated here.
        sem_gradient = self._analytic_error_gradient(
            sem_probabilities, consensus, self.sem_head
        )
        if self.gradient_detach:
            sem_gradient = sem_gradient.detach()
        auxiliary_feature = sem_gradient.square()
        auxiliary_feature = auxiliary_feature / auxiliary_feature.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.eps)
        auxiliary_feature = self.auxiliary_projection(auxiliary_feature)

        projected_syn = F.softmax(self.syn_projection(syn_features), dim=-1)
        projected_sem = F.softmax(self.sem_projection(sem_features), dim=-1)
        fused = self.output_norm(
            self.alpha * projected_syn
            + self.beta * projected_sem
            + gate * auxiliary_feature
        )

        auxiliary_loss = syn_logits.sum() * 0.0
        if labels is not None:
            class_ids, targets = self._label_targets(labels, syn.device)
            if class_ids.size(0) != syn.size(0):
                raise ValueError("AGF label count differs from feature batch size")
            targets = targets.to(dtype=syn_probabilities.dtype)
            # Labels are used only after `fused` has been computed, for training
            # supervision; changing them cannot change this forward's prediction.
            true_syn_error = (
                syn_probabilities - targets
            ).square().sum(dim=-1, keepdim=True).sqrt()
            true_sem_error = (
                sem_probabilities - targets
            ).square().sum(dim=-1, keepdim=True).sqrt()
            syn_error_target = self._normalize_error_targets(
                true_syn_error.detach()
            )
            sem_error_target = self._normalize_error_targets(
                true_sem_error.detach()
            )
            branch_loss = 0.5 * (
                F.cross_entropy(syn_logits, class_ids)
                + F.cross_entropy(sem_logits, class_ids)
            )
            reliability_loss = 0.5 * (
                F.mse_loss(syn_error, syn_error_target)
                + F.mse_loss(sem_error, sem_error_target)
            )
            auxiliary_loss = self.aux_loss_weight * (
                branch_loss + reliability_loss
            )

        return {
            "fused": fused,
            "syn_logits": syn_logits,
            "sem_logits": sem_logits,
            "auxiliary_loss": auxiliary_loss,
            "gate": gate,
            "syn_error": syn_error,
            "sem_error": sem_error,
        }
