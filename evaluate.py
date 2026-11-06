"""Evaluate a full AEOR+RG checkpoint, including legacy frozen-v1+RG weights."""
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from aeor_rg.runtime import ROOT, resolve_backbone


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing outputs are never overwritten")
    parser.add_argument("--roberta-path", type=Path)
    parser.add_argument("--lexicon-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--export-routes", action="store_true")
    parser.add_argument("--latency-sentences", type=int, default=0)
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 0 or args.latency_sentences < 0 or args.latency_repeats < 1:
        parser.error("Invalid evaluation settings")
    if args.output.exists():
        parser.error("--output must be a new directory")
    import torch
    from transformers import RobertaTokenizerFast
    from innovations.paper_experiments import VARIANT as PAPER_VARIANT
    from innovations.paper_experiments.model import PaperAblationDAGF
    from innovations.paper_experiments.train import validate_checkpoint as validate_paper
    from innovations.v1_relation_gate import VARIANT as LEGACY_VARIANT
    from innovations.v1_relation_gate.model import V1RelationGateDAGF
    from innovations.v1_relation_gate.train import validate_checkpoint as validate_legacy, dataset_options, _check_device
    from innovations.v1_relation_gate.evaluation import evaluate, benchmark_latency, _dataset_class
    from aeor_rg.runtime import write_json
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, kwargs = dict(checkpoint["config"]), dict(checkpoint["model_kwargs"])
    if checkpoint.get("variant") == PAPER_VARIANT:
        validate_paper(checkpoint)
        if config.get("comparison_profile") != "full":
            raise ValueError("Use a full-model checkpoint, not an ablation")
        factory = PaperAblationDAGF
    elif checkpoint.get("variant") == LEGACY_VARIANT:
        validate_legacy(checkpoint)
        if config.get("comparison_profile") != "v1_relation_gate":
            raise ValueError("Legacy checkpoint is not frozen v1+RG")
        factory = V1RelationGateDAGF
    else:
        raise ValueError("Unsupported checkpoint variant")
    for key in ("use_routing", "use_inter_aspect", "use_structure_prior", "use_amr", "use_agf", "relation_reweight"):
        if kwargs.get(key) is not True:
            raise ValueError("Full-model checkpoint disables " + key)
    if kwargs.get("use_shared_channel", True) is not True or config.get("no_lexicon", False):
        raise ValueError("Full model requires shared channel and lexical graph features")
    if kwargs.get("readout") != "cls" or kwargs.get("baseline_regularization_weight") != 0:
        raise ValueError("Readout/objective differs from the released full model")
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = _check_device(args.device)
    original_backbone = Path(config["roberta_path"])
    backbone = args.roberta_path or (original_backbone if original_backbone.is_dir() else ROOT / "roberta")
    config["roberta_path"] = kwargs["roberta_path"] = str(resolve_backbone(backbone))
    original_lexicon = Path(config["lexicon_dir"])
    config["lexicon_dir"] = str((args.lexicon_dir or (original_lexicon if original_lexicon.is_dir() else ROOT / "opinion-lexicon-English")).resolve())
    tokenizer = RobertaTokenizerFast.from_pretrained(config["roberta_path"], add_prefix_space=True, local_files_only=True)
    options = dataset_options(config, tokenizer, checkpoint["amr_vocab"], checkpoint["dependency_vocab"])
    dataset = _dataset_class(config["dataset"])(args.data_file, require_labels=False, **options)
    model = factory(**kwargs).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    report, predictions = evaluate(model, dataset, device, args.batch_size, include_routes=args.export_routes)
    report.update(data_stats=dataset.stats, model_variant=config["model_variant"], readout="cls",
                  source_checkpoint=str(args.checkpoint.resolve()))
    report["latency"] = (benchmark_latency(model, dataset, device, repeats=args.latency_repeats,
                                           max_sentences=args.latency_sentences)
                         if args.latency_sentences else dict(status="not_evaluated"))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "report.json", report)
    write_json(args.output / "predictions.json", predictions)
    print("Saved evaluation: " + str(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
