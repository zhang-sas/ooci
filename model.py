"""Convenient imports; AEORRG keeps the original state-dict parameter names.

Construct with a full checkpoint's model_kwargs, or use train.py. The alias
alone does not supply dataset-specific dependency vocabulary sizes.
"""
from innovations.paper_experiments.model import PaperAblationDAGF as AEORRG
from innovations.v1_reference.model import OpinionRouter, SharedEvidenceInteraction
from innovations.v1_relation_gate.layers import RelationReweightedGCN

__all__ = ["AEORRG", "OpinionRouter", "SharedEvidenceInteraction", "RelationReweightedGCN"]
