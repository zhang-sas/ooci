"""Use the unchanged v1 optimizer/protocol with independent ablation metadata."""
from . import VARIANT
from .model import PaperAblationDAGF
from .spec import PROFILES
from ..v1_relation_gate import train as gate_train
from ..typed_stability.suite import write_json, read_json


def build_parser():
    parser = gate_train.build_parser()
    parser.description = __doc__
    # This parser object belongs to this call, not to the old training entry.
    for action in parser._actions:
        if action.dest == "comparison_profile":
            action.choices = tuple(PROFILES)
            action.required = True
    parser.add_argument("--no-shared", action="store_true")
    return parser


def run(args):
    if args.evaluation_protocol != "baseline" or args.val_file is not None:
        raise ValueError("Use full train and periodic test selection")
    if args.readout != "cls" or args.baseline_regularization_weight != 0:
        raise ValueError("Retain v1 CLS readout and baseline_regularization_weight=0")
    for name, expected in PROFILES[args.comparison_profile].items():
        if getattr(args, name) is not expected:
            raise ValueError("Profile/flag mismatch: " + name)
    args.eval_interval_unit = "optimizer_step"
    args.additive_relation_messages = False
    args.base_model = "frozen_v1"
    args.dependency_feature_usage = "edge_gate_only" if args.relation_reweight else "none"
    args.paper_ablation_schema = VARIANT
    extras = {k: getattr(args, k) for k in
              ("relation_reweight", "gate_relation_dim", "gate_hidden_dim", "gate_strength")}
    extras["use_shared_channel"] = not args.no_shared
    output = gate_train._run(args, variant=VARIANT, model_factory=PaperAblationDAGF, extra_model_kwargs=extras)
    completion = read_json(output / "completion.json")
    completion.update(comparison_profile=args.comparison_profile, dataset=args.dataset,
                      paper_ablation_schema=VARIANT)
    write_json(output / "completion.json", completion)
    return output


def main(argv=None):
    return run(build_parser().parse_args(argv))


def validate_checkpoint(checkpoint):
    gate_train.validate_checkpoint(checkpoint)
    config, kwargs = checkpoint["config"], checkpoint["model_kwargs"]
    if checkpoint.get("variant") != VARIANT:
        raise ValueError("Wrong paper-ablation checkpoint variant")
    profile = config.get("comparison_profile")
    if profile not in PROFILES or config.get("paper_ablation_schema") != VARIANT:
        raise ValueError("Missing independent ablation metadata")
    for name, expected in PROFILES[profile].items():
        if config.get(name) is not expected:
            raise ValueError("Checkpoint profile mismatch: " + name)
    for name, expected in (("use_routing", not config["no_routing"]),
                           ("use_inter_aspect", not config["no_inter_aspect"]),
                           ("use_shared_channel", not config["no_shared"])):
        if kwargs.get(name) is not expected:
            raise ValueError("Checkpoint model option mismatch: " + name)

