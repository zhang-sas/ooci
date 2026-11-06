"""Reweight original GCN content messages; relation embeddings only score edges."""
import math
import torch
from torch import nn
from torch.nn import functional as F

from ..v1_reference.baseline import load_baseline_module
from ..typed_syntax.layers import TypedSyntaxGCN

DenseSyntaxGCN=load_baseline_module("model").DenseSyntaxGCN


class RelationReweightedGCN(DenseSyntaxGCN):
    """Original GCN with bounded relation/direction edge gates; no additive branch."""
    _validated_edges=TypedSyntaxGCN._validated_edges

    def __init__(self,dim,dropout,num_relations,relation_dim=32,hidden_dim=32,strength=.5):
        for name,value in (("num_relations",num_relations),("relation_dim",relation_dim),("hidden_dim",hidden_dim)):
            if isinstance(value,bool) or not isinstance(value,int) or value<1:
                raise ValueError(name+" must be a positive integer")
        if isinstance(strength,bool) or not math.isfinite(strength) or not 0<=strength<1:
            raise ValueError("Gate strength must be finite and in [0,1)")
        super().__init__(dim,dropout)
        self.num_relations=num_relations
        self.strength=float(strength)
        self.relation_dim=relation_dim
        self.gate_hidden_dim=hidden_dim
        with torch.random.fork_rng(devices=[]):
            self.gate_relation_embedding=nn.Embedding(num_relations,relation_dim,padding_idx=0)
            self.gate_direction_embedding=nn.Embedding(4,relation_dim)
            nn.init.normal_(self.gate_relation_embedding.weight,std=.02)
            nn.init.normal_(self.gate_direction_embedding.weight,std=.02)
            with torch.no_grad():self.gate_relation_embedding.weight[0].zero_()
            self.gate_hidden=nn.Linear(2*relation_dim,hidden_dim)
            self.gate_score=nn.Linear(hidden_dim,1,bias=False)
            nn.init.zeros_(self.gate_score.weight)

    @classmethod
    def from_base(cls,base,num_relations,relation_dim=32,hidden_dim=32,strength=.5):
        if not isinstance(base,DenseSyntaxGCN):raise TypeError("Expected original DenseSyntaxGCN")
        with torch.random.fork_rng(devices=[]):
            layer=cls(base.message.in_features,base.dropout.p,num_relations,relation_dim,hidden_dim,strength)
        for name in ("message","self_projection","norm","dropout"):
            setattr(layer,name,getattr(base,name))
        layer.to(device=base.message.weight.device,dtype=base.message.weight.dtype)
        layer.train(base.training)
        return layer

    def sparse_gates(self,types,directions,src,dst,active):
        features=torch.cat((self.gate_relation_embedding(types),self.gate_direction_embedding(directions)),dim=-1)
        delta=self.strength*torch.tanh(self.gate_score(F.gelu(self.gate_hidden(features))).squeeze(-1))
        return 1+torch.where(active & src.ne(dst),delta,torch.zeros_like(delta))

    def forward(self,states,adjacency,node_mask,dependency_edges=None):
        if self.strength==0:return super().forward(states,adjacency,node_mask)
        if states.ndim!=3 or adjacency.shape!=(states.size(0),states.size(1),states.size(1)):
            raise ValueError("Expected states [B,L,D] and adjacency [B,L,L]")
        if node_mask.shape!=states.shape[:2] or node_mask.dtype!=torch.bool:
            raise ValueError("Expected boolean node mask [B,L]")
        mask=node_mask.to(states.dtype)
        pairs=mask.unsqueeze(2)*mask.unsqueeze(1)
        graph=adjacency.to(states.dtype).clamp_min(0.)*pairs
        identity=torch.eye(states.size(1),device=states.device,dtype=states.dtype)
        graph=graph+identity.unsqueeze(0)*pairs
        graph=graph/graph.sum(-1,keepdim=True).clamp_min(1.)
        src,dst,types,directions,active=self._validated_edges(dependency_edges,states,graph)
        gates=self.sparse_gates(types,directions,src,dst,active)
        length=states.size(1)
        # Padding writes only zero to diagonal slot zero, never to a non-self edge.
        codes=torch.where(active,dst*length+src,torch.zeros_like(src))
        delta=graph.new_zeros((states.size(0),length*length)).scatter(1,codes,gates-1).view_as(graph)
        correction=graph*delta
        graph=(graph+correction)/(1+correction.sum(-1,keepdim=True))
        update=torch.bmm(graph,self.message(states))+self.self_projection(states)
        result=self.norm(states+self.dropout(F.gelu(update)))
        return result*mask.unsqueeze(-1)

    def diagnostics(self):
        return dict(enabled=True,mode="relation_only",additive_relation_message=False,strength=self.strength,
            gate_relation_dim=self.relation_dim,gate_hidden_dim=self.gate_hidden_dim,
            gate_score_norm=float(self.gate_score.weight.detach().norm().cpu()),
            existing_edges_only=True,self_edge_gates_fixed=True)
