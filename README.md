# Toxic Comment Classification (NLP 2026)

Multi-label classification on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) dataset. Given a comment, predict six toxicity labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

This repository implements:

1. **Baseline** — TF-IDF (1–2 grams) + one-vs-rest logistic regression  
2. **DistilBERT** — `distilbert-base-uncased` fine-tuned with Hugging Face `Trainer`  
3. **HateBERT** — `GroNLP/hateBERT` (abusive-language pre-trained BERT) fine-tuned the same way  

Training uses a fixed validation split (`random_state=42`, 10% holdout), reports ROC-AUC / PR-AUC / F1, and writes timestamped Kaggle submissions and optional figures under `figs/`.

## Requirements

- Python 3.10+  
- GPU recommended for transformer training  
- Dependencies: `pip install -r requirements.txt`  

Additional packages used at runtime: `kagglehub` (competition data), `huggingface_hub` (usually installed with `transformers`).

## Quick start

```bash
cd project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install kagglehub
```

### 1. Download competition data

```bash
export KAGGLE_API_TOKEN="your_token"
python download.py
```

Files are extracted to `datasets/` (`train.csv`, `test.csv`, `sample_submission.csv` or `.csv.zip`).

### 2. (Optional) Download HateBERT weights locally

If Hugging Face downloads are slow or time out:

```bash
export HF_ENDPOINT=https://hf-mirror.com   # optional mirror
export HF_HUB_DISABLE_XET=1
python download_model.py GroNLP/hateBERT
```

Weights land in `models/hateBERT/`. `train_hatebert.py` uses this folder automatically when `model.safetensors` is present.

### 3. Train models

**Baseline** (CPU-friendly):

```bash
python baseline_tfidf_lr.py
```

**DistilBERT** (default batch size 16):

```bash
python train_distilbert.py
```

**HateBERT** (default batch size 8; use local model if downloaded):

```bash
python train_hatebert.py
# or explicitly:
python train_hatebert.py --model-name ./models/hateBERT
```

**Debug run** (small subset, 1 epoch):

```bash
python train_hatebert.py --max-train-samples 2000 --epochs 1
```

### 4. Outputs

| Output | Location |
|--------|----------|
| Submission CSV | `submission_<model>_<YYYYMMDD_HHMMSS>.csv` |
| Checkpoints | `checkpoints/distilbert/`, `checkpoints/hatebert/` |
| Figures | `figs/<model>_<timestamp>/` (unless `--no-figures`) |

Large artifacts (`datasets/`, `models/`, `checkpoints/`, `figs/`, submissions) are **not** tracked in git; see `.gitignore`.

## Training details

- **Validation**: default `--val-size 0.1`, then refit on full training data (same as baseline).  
- **Epochs**: default `--epochs 2` per training phase (validation run + full-data run).  
- **Transformer loss**: multi-label BCE with sigmoid (`problem_type="multi_label_classification"`).  
- **Metrics on validation**: per-label ROC-AUC, mean ROC-AUC, PR-AUC, macro/micro F1.  

### Useful flags

```bash
# Skip plots
python train_distilbert.py --no-figures

# Train once on full data only (no validation phase)
python train_transformer.py --val-size 0

# OOM on GPU
python train_hatebert.py --batch-size 4 --max-length 128
```

## Figures

With figures enabled (default), outputs go to `figs/<model>_<timestamp>/`:

| Model | Plots |
|-------|--------|
| **Baseline** | Confusion matrices, ROC curves, top TF-IDF terms per label |
| **Transformers** | Loss / metric / LR curves, confusion matrices, ROC & PR curves, per-label metric bars, score histograms, calibration curves, label frequency & co-occurrence |

Baseline does **not** plot training loss curves (single `fit` on sklearn pipeline).

## Project layout

```
.
├── baseline_tfidf_lr.py      # TF-IDF + logistic regression
├── train_transformer.py    # Shared Trainer pipeline
├── train_distilbert.py     # Entry: DistilBERT defaults
├── train_hatebert.py       # Entry: HateBERT defaults
├── download.py             # Kaggle competition data
├── download_model.py         # Hugging Face model snapshot (e.g. HateBERT)
├── run_paths.py            # Timestamped paths for submissions & figs
├── viz.py                  # Plotting helpers
├── requirements.txt
├── datasets/               # (local, gitignored)
├── models/                 # (local, gitignored)
├── checkpoints/            # (local, gitignored)
└── figs/                   # (local, gitignored)
```

## Citation / models

- **DistilBERT**: [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)  
- **HateBERT**: [GroNLP/hateBERT](https://huggingface.co/GroNLP/hateBERT) — Caselli et al., WOAH 2021  
- **Data**: Jigsaw Toxic Comment Classification Challenge (Kaggle)

## License

Course project (NLP 2026). Check Kaggle competition rules and Hugging Face model licenses before redistributing data or weights.
