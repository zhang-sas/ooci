"""Low-dimensional dependency-type/direction messages on the unchanged graph.

The base message, self projection, residual, nonlinearity, dropout and norm are
retained. Every sparse edge indexes graph[receiver, sender], after exactly the
base layer's self-loop addition and normalization. No dense [B,L,L,D] tensor is
created. Direction zero is a real SELF direction, not an embedding padding id.
"""
import torch
from torch import nn
from torch.nn import functional as F

from ..v1_reference.baseline import load_baseline_module

DenseSyntaxGCN = load_baseline_module("model").DenseSyntaxGCN
FEATURE_MODES = ("none", "type", "direction", "both")
EDGE_KEYS = ("dep_src", "dep_dst", "dep_type", "dep_direction", "dep_edge_mask")


class TypedSyntaxGCN(DenseSyntaxGCN):
    """A base GCN plus a zero-initialized, edge-typed additive message."""
    def __init__(self, dim, dropout, num_relations, relation_dim=32, feature_mode="both"):
        if feature_mode not in FEATURE_MODES:
            raise ValueError("Unknown typed syntax feature_mode: " + str(feature_mode))
        for name, value in (("num_relations", num_relations), ("relation_dim", relation_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(name + " must be a positive integer")
        super().__init__(dim, dropout)
        self.num_relations = num_relations
        self.relation_dim = relation_dim
        self.feature_mode = feature_mode
        # New CPU-created parameters do not change the subsequent base RNG
        # stream. The zero output projection gives a literal base-layer start.
        with torch.random.fork_rng(devices=[]):
            dimensions = 0
            if feature_mode in ("type", "both"):
                self.relation_embedding = nn.Embedding(num_relations, relation_dim, padding_idx=0)
                nn.init.normal_(self.relation_embedding.weight, std=.02)
                with torch.no_grad():
                    self.relation_embedding.weight[0].zero_()
                dimensions += relation_dim
            if feature_mode in ("direction", "both"):
                self.direction_embedding = nn.Embedding(4, relation_dim)
                nn.init.normal_(self.direction_embedding.weight, std=.02)
                dimensions += relation_dim
            if dimensions:
                self.relation_projection = nn.Linear(dimensions, dim, bias=False)
                nn.init.zeros_(self.relation_projection.weight)

    @classmethod
    def from_base(cls, base_layer, num_relations, relation_dim=32, feature_mode="both"):
        """Reuse the original base submodules and parameter names, without RNG drift."""
        if not isinstance(base_layer, DenseSyntaxGCN):
            raise TypeError("from_base requires the unchanged DenseSyntaxGCN")
        with torch.random.fork_rng(devices=[]):
            result = cls(base_layer.message.in_features, base_layer.dropout.p,
                         num_relations, relation_dim, feature_mode)
        # Reuse objects rather than loading copies: pretrained/base parameters
        # keep their identity, values, dtype, device, and optimizer-facing names.
        result.message = base_layer.message
        result.self_projection = base_layer.self_projection
        result.norm = base_layer.norm
        result.dropout = base_layer.dropout
        result.to(device=base_layer.message.weight.device, dtype=base_layer.message.weight.dtype)
        result.training = base_layer.training
        for name, module in result.named_children():
            if name not in ("message", "self_projection", "norm", "dropout"):
                module.train(base_layer.training)
        return result

    def _validated_edges(self, edges, states, normalized_graph):
        if not isinstance(edges, dict) or any(name not in edges for name in EDGE_KEYS):
            raise ValueError("Typed syntax requires all dep edge tensors")
        src, dst, types, directions, active = (edges[name] for name in EDGE_KEYS)
        if src.ndim != 2 or src.size(0) != states.size(0):
            raise ValueError("Dependency edge tensors must have shape [B,E]")
        if any(value.shape != src.shape for value in (dst, types, directions, active)):
            raise ValueError("Dependency edge shapes do not match")
        if any(value.device != states.device for value in (src, dst, types, directions, active)):
            raise ValueError("Dependency edges and states must share a device")
        if any(value.dtype != torch.long for value in (src, dst, types, directions)) or active.dtype != torch.bool:
            raise ValueError("Dependency indices/types/directions must be long and edge mask bool")
        length = states.size(1)
        if (((src < 0) | (src >= length) | (dst < 0) | (dst >= length)) & active).any():
            raise ValueError("Active dependency node index outside sequence")
        if (((types <= 0) | (types >= self.num_relations)) & active).any():
            raise ValueError("Active dependency type must be a non-padding vocabulary id")
        if (((directions < 0) | (directions >= 4)) & active).any():
            raise ValueError("Active dependency direction outside 0..3")
        src, dst = src.clamp(0, length - 1), dst.clamp(0, length - 1)
        # Integer counts validate one edge per nonzero normalized adjacency
        # entry. Padded duplicates contribute zero and cannot overwrite an edge.
        codes = dst * length + src
        counts = torch.zeros((states.size(0), length * length), dtype=torch.long, device=states.device)
        counts.scatter_add_(1, codes, active.long())
        if not torch.equal(counts.view_as(normalized_graph), normalized_graph.gt(0).long()):
            raise ValueError("Dependency edges must uniquely cover the unchanged positive graph")
        # Invalid padded IDs are never looked up and never receive gradients.
        safe_types = torch.where(active, types, torch.zeros_like(types))
        safe_directions = torch.where(active, directions, torch.zeros_like(directions))
        return src, dst, safe_types, safe_directions, active

    def forward(self, states, adjacency, node_mask, typed_edges=None):
        if self.feature_mode == "none":
            return super().forward(states, adjacency, node_mask)
        if states.ndim != 3 or adjacency.shape != (states.size(0), states.size(1), states.size(1)):
            raise ValueError("Expected states [B,L,D] and adjacency [B,L,L]")
        if node_mask.shape != states.shape[:2] or node_mask.dtype != torch.bool:
            raise ValueError("Expected boolean node_mask [B,L]")
        if adjacency.device != states.device or node_mask.device != states.device:
            raise ValueError("States, graph and node mask must share a device")
        mask = node_mask.to(states.dtype)
        valid_pairs = mask.unsqueeze(2) * mask.unsqueeze(1)
        graph = adjacency.to(states.dtype).clamp_min(0.0) * valid_pairs
        identity = torch.eye(states.size(1), device=states.device, dtype=states.dtype)
        graph = graph + identity.unsqueeze(0) * valid_pairs
        graph = graph / graph.sum(dim=-1, keepdim=True).clamp_min(1.0)
        src, dst, types, directions, active = self._validated_edges(typed_edges, states, graph)
        batch = torch.arange(states.size(0), device=states.device).unsqueeze(1)
        weights = graph[batch, dst, src] * active.to(states.dtype)
        features = []
        if self.feature_mode in ("type", "both"):
            features.append(self.relation_embedding(types))
        if self.feature_mode in ("direction", "both"):
            features.append(self.direction_embedding(directions))
        relations = torch.cat(features, dim=-1) if len(features) > 1 else features[0]
        # [B,L,E] assignment and [B,E,d_r] messages: no edge-width D tensor or
        # floating scatter/atomic aggregation. Use bmm for the low-rank sum.
        assignment = torch.arange(states.size(1), device=states.device).view(1, -1, 1).eq(dst.unsqueeze(1))
        summary = torch.bmm(assignment.to(states.dtype), relations.to(states.dtype) * weights.unsqueeze(-1))
        update = torch.bmm(graph, self.message(states)) + self.self_projection(states)
        update = update + self.relation_projection(summary)
        result = self.norm(states + self.dropout(F.gelu(update)))
        return result * mask.unsqueeze(-1)
