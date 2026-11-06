"""Public experiment definitions, kept importable without torch."""
PROFILES = {
    "no_route": dict(no_routing=True, relation_reweight=False, no_shared=False, no_inter_aspect=False),
    "v1": dict(no_routing=False, relation_reweight=False, no_shared=False, no_inter_aspect=False),
    "no_route_rg": dict(no_routing=True, relation_reweight=True, no_shared=False, no_inter_aspect=False),
    "full": dict(no_routing=False, relation_reweight=True, no_shared=False, no_inter_aspect=False),
    "no_shared_rg": dict(no_routing=False, relation_reweight=True, no_shared=True, no_inter_aspect=False),
    "no_inter_rg": dict(no_routing=False, relation_reweight=True, no_shared=False, no_inter_aspect=True),
}

