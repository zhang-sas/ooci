# OOCI

**Opinion Ownership and Conflict Isolation**

OOCI is a framework for multi-aspect sentiment analysis. It reuses sentence encodings and separates aspect-exclusive and shared evidence to reduce cross-aspect sentiment interference while preserving useful sharing.

## Method

- Allocate token evidence to aspect-exclusive, shared, and background channels.
- Use independent membership gates to select which aspects can access shared evidence.
- Read exclusive and shared evidence separately, and regulate aspect interaction through shared-evidence overlap.

## Requirements

- `torch==1.4.0`
- `scikit-learn==0.23.2`
- `transformers==4.41.0`
- `einops==0.7.0`
- `cython==0.29.13`
- `nltk==3.5`
- `numpy==1.20.3`
- `spacy==3.7.4`
- `benepar==0.2.0`

To install requirements, run:

```bash
pip install -r requirements.txt
```

## Preparation

Prepare the MAMS and SemEval-2014 Restaurant and Laptop datasets, then run:

```bash
python dataset/preprocess_data.py
python dataset/new_perocess_data.py
```

Obtain `embedding.pt` and `stoi.pt` from [APARN](https://github.com/THU-BPM/APARN) and place them in the project root.

Place RoBERTa-base weights and tokenizer files in `roberta/` (or `roberta/roberta/`), the English opinion lexicon in `opinion-lexicon-English/`, and dependency- and AMR-annotated data in `dataset/` as `train_with_amr.json` and `test_with_amr.json` under each dataset directory.

## Training

Run the full model on all three datasets with five random seeds:

```bash
bash run_full_model_5seeds.sh all
```

Seeds are `1000–1004`. Replace `all` with `restaurant`, `laptop`, or `mams` to run a single dataset.

Results are saved under `runs_full_model_5seeds/`, with summaries in `mean_std.csv` and `summary.md`. See the [reproduction guide (Chinese)](README_FULL_MODEL_5SEEDS_20261104.md) for details.

## Main Results

Paper-reported means over five seeds (%).

| Dataset | Accuracy | Macro-F1 |
|---|---:|---:|
| MAMS | 86.30 | 85.81 |
| SemEval-2014 Restaurant | 89.92 | 85.12 |
| SemEval-2014 Laptop | 85.13 | 82.48 |

The default baseline evaluation protocol uses the test set for checkpoint selection. Accuracy and Macro-F1 are recorded at their respective best checkpoints.
