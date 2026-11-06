# DAGF-AEOR-RG

Standalone source release of the project's **full AEOR + relation-only RG model**.
See [中文说明](README.md) for detailed commands and architecture locations.

This release copies the frozen v1 router, shared-evidence interaction, DAGF core,
RG layers and training backend without modifying their source bytes. Compatibility
module names are retained so existing full-model checkpoints keep their parameter names.
It excludes separate v3, scope, multilayer, TAPT and mechanism-control experiments.

## Setup

Use the existing working Python 3.8/3.10 DAGF environment or install `requirements.txt`.
The pinned dependencies are inherited from the project, not a newly benchmarked environment.
For GPU installation, install the matching PyTorch wheel first.

Provide the **same processed corpora, RoBERTa checkpoint/tokenizer and opinion lexicon**
used in the experiments. These assets are excluded from this repository.
Input locations and schema are documented in [docs/DATA.md](docs/DATA.md).
The repository consumes preprocessed, word-aligned dependency/AMR JSON files;
it does not bundle an end-to-end raw-text parsing pipeline.

```bash
python -B verify_release.py
python -B scripts/check_inputs.py all
bash run.sh restaurant 1000
bash run.sh laptop 1000
bash run.sh mams 1000
bash run.sh twitter 1000
```

All public training commands retain routing, sharing, inter-aspect interaction,
RG, CLS readout and `baseline_regularization_weight=0`. No opinion-span or
shared-membership supervision is added. Test polarity never defines graph edges.

## Seeds and evaluation

```bash
# Restaurant, Laptop and MAMS, five independent seeds each, sequentially on one GPU
bash run_seeds.sh all
# Twitter, five independent seeds
bash run_seeds.sh twitter
# Print only; no model execution or run creation
bash run_seeds.sh all --dry-run
```

Seeds default to 1000–1004. Completed jobs are verified and skipped. Interrupted
jobs restart in a new attempt rather than resuming optimizer state. Reports include
per-seed metrics and sample standard deviation (ddof=1); incomplete seed groups
do not receive a complete-group mean/SD.

**Evaluation protocol:** preserve the existing full-training-set / periodic-test
checkpoint selection protocol, including epoch-end tests. No validation split is
introduced. This is test selection, not validation-based model selection.
Maximum Acc and maximum Macro-F1 are reported independently. `best.pt` and the
default final report use `best_acc.pt`; evaluate `best_f1.pt` for metrics paired
with the F1 maximum.

```bash
python -B evaluate.py \
  --checkpoint /path/to/completed/run/best_f1.pt \
  --data-file dataset/Restaurants_corenlp/test_with_amr.json \
  --output runs_eval/restaurant_best_f1 --device cuda --export-routes
```

The evaluator accepts both the current paper-full variant and legacy frozen
v1+RG checkpoints, checks the full-model flags and uses strict state-dict loading.
Use a new output directory. Relocated assets may be supplied using
`--roberta-path` and `--lexicon-dir`.

No benchmark results or claims of verified evidence ownership are added by this
packaging step. The latent routing and shared-member gates require separate
mechanism evaluation.

## Provenance

`source_provenance.json` records the unchanged source-file hashes.
`release_manifest.json` records the complete release inventory.
`verify_release.py` checks hashes, Python 3.8 syntax and local imports without
requiring PyTorch or executing a model. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for the current licensing/provenance status.
