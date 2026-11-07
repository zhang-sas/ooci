# OOCI

**Opinion Ownership and Conflict Isolation**

OOCI is a framework for multi-aspect sentiment analysis. It reuses sentence encodings and allocates token evidence to aspect-exclusive, shared, and background channels. Membership gates control each aspect's access to shared evidence, and shared-evidence overlap regulates interactions between aspects. The full model also uses AMR graphs, auxiliary graph fusion, and dependency-relation edge gating.

## Requirements

Install the dependencies listed in `requirements.txt` in your Python environment. The default configuration uses CUDA.

## Data and Model Preparation

Prepare the datasets, RoBERTa-base weights, and English opinion lexicon separately.

| Dataset | Default data directory | Source |
| --- | --- | --- |
| `restaurant` | `dataset/Restaurants_corenlp/` | [SemEval-2014 Task 4](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools) |
| `laptop` | `dataset/Laptops_corenlp/` | [SemEval-2014 Task 4](https://alt.qcri.org/semeval2014/task4/index.php?id=data-and-tools) |
| `mams` | `dataset/MAMS_corenlp/` | [MAMS-ATSA](https://github.com/siat-nlp/MAMS-for-ABSA/tree/master/data/MAMS-ATSA/raw) |

Create each data directory and place its `train_with_amr.json` and `test_with_amr.json` files there. Use data with dependency parses and token-aligned AMR annotations.

- Each file contains a list of records with `token`, `head`, `deprel`, `amr_tokens`, `amr_edges`, and `aspects`.
- `head` and `deprel` have the same length as `token`. Dependency heads use 1-based indices with 0 for the root. AMR edges have the form `[source, relation, target]` with 0-based indices into `amr_tokens`. Both tokenizations must represent the same text.
- Each aspect contains `from`, `to`, `term`, and `polarity`. The span uses `[from, to)` indices into `token`, and `term` is the corresponding list of words. Training and test polarities are `positive`, `negative`, or `neutral`.
- Download the RoBERTa-base `config.json`, `vocab.json`, `merges.txt`, and model weights and place them in `roberta/`. The nested layout `roberta/roberta/` is also supported. [RoBERTa-base files](https://huggingface.co/FacebookAI/roberta-base/tree/main)
- Download and extract the English opinion lexicon, then place `positive-words.txt` and `negative-words.txt` in `opinion-lexicon-English/`. [English opinion lexicon](https://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar)

AMR-related data and reference code are available from APARN. The full OOCI model learns its AMR relation embeddings from the training data. APARN's original path-embedding assets are also linked for reference. [APARN](https://github.com/THU-BPM/APARN), [embedding.pt](https://github.com/THU-BPM/APARN/blob/main/APARN/embedding.pt), [stoi.pt](https://github.com/THU-BPM/APARN/blob/main/APARN/stoi.pt)

## Training

Run one full-model experiment from the repository root.

```bash
python -B train.py --dataset restaurant --seed 1000
```


## Reported Results

Paper-reported means over five seeds, in percent.

| Dataset | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| MAMS | 86.30 | 85.81 |
| SemEval-2014 Restaurant | 89.92 | 85.12 |
| SemEval-2014 Laptop | 85.13 | 82.48 |

Source provenance and third-party asset information are provided in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
