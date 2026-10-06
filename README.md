# DeepAptamer
DeepAptamer: SELEX-based Aptamer Affinity Prediction by Hybrid Deep Learning
![1-s2 0-S2162253124003238-fx1_lrg](https://github.com/user-attachments/assets/68083cd3-62a5-4db7-b38c-ce6c4fec0e3f)

DeepAptamer classifies 35-nt aptamer sequences as high-affinity binders or non-binders for one
target protein. It learns from SELEX sequencing data: the most enriched sequences in the final
round count as binders and the least enriched as non-binders. Each sequence is described by a
35×4 one-hot encoding plus 126 DNA shape values (MGW, ProT, Roll, HelT from DNAshapeR).

The `deepaptamer/` package is a runnable version of the original research scripts in `script/`.
It reproduces the published results from the shipped weights (see [Reproduction](#reproduction)).

## Setup

Python only — no R needed.

```bash
conda env create -f environment.yml && conda activate deepaptamer
```

or `pip install -r requirements.txt` into any Python 3.11 environment.
Run every command from this folder.

## The app

```bash
streamlit run app.py --server.address localhost
```

Opens at http://localhost:8501 with three tabs:

1. **Prepare reads** — upload FASTQ, FASTA or a sequence list (`.gz` fine), give the primer
   sequences, and it trims them, checks lengths, collapses duplicates into counts and shows you
   what was dropped. Reverse-strand reads are matched automatically.
2. **Score sequences** — pick a supplied target or a model you trained, and score. Encoding
   runs in blocks, so memory stays flat however long the library is.
3. **Train a new target** — set the binder and non-binder counts, watch per-epoch progress,
   and get held-out metrics with ROC and precision-recall curves. Trained models are saved to
   `runs/<name>/` and appear in the scoring tab.

Pass `--server.address localhost` as above unless you intend otherwise: Streamlit binds to all
interfaces by default, which puts an upload form on your network.

Everything the app does is also available from the command line, below.

### Sharing it with colleagues

Push this repo to GitHub and point [share.streamlit.io](https://share.streamlit.io) at `app.py`.
Community Cloud installs `requirements.txt` automatically. Two things to expect there:

- **Scoring works well.** Encoding is pure numpy and the models are small.
- **Training is better done locally.** A full run takes tens of minutes and the free tier will
  usually recycle the app first. Train on your own machine and commit the `runs/<name>/` folder;
  it then appears in the app's model list for everyone.

Anyone with the link can use a public Community Cloud app, so keep unpublished sequences off it.

## Score your own sequences

Put your sequences in a file, one per line, or as FASTA. Every sequence must be exactly 35 nt
of A/C/G/T. Pad shorter reads with trailing `N`, as the training data does.

```bash
python -m deepaptamer predict --target CTGF --input my_seqs.txt -o predictions.tsv
```

The output has one score per model: the probability that the sequence is a high-affinity binder
for that target. The shipped weights only cover the targets in the table below. For any other
protein, train on your own SELEX data first.

| `--target` | Protein | Weights |
|---|---|---|
| `CTGF` | Connective tissue growth factor | `model/CTGF/9k_1w` |
| `BCMA` | B-cell maturation antigen | `model/BCMA/4k_7k/best_model` |

## Train on your own SELEX data

### Preparing the file

Give a tab-separated file with one sequence per row and its read count (or frequency) in
column 2. Rows don't need to be sorted. To build it from your final-round reads:

1. **Trim the constant primer regions**, leaving only the randomised region.
2. **Keep only 35-nt reads**, or pad shorter ones with trailing `N`. Both the architecture and
   the shape features are fixed at 35 nt.
3. **Collapse duplicates and count** them, e.g. `sort | uniq -c`.

The app's first tab does all three for you.

Your file needs both ends of the enrichment range from the same experiment. In the shipped
data the non-binders are singleton reads — sequences seen exactly once in the final round —
and they differ from the enriched ones in base composition, which is the signal the models
learn. A file of only enriched sequences cannot be trained on.

### Running it

`--pos` sets how many of the most enriched sequences become binders, `--neg` how many of the
least enriched become non-binders. Everything in between is discarded.

```bash
python -m deepaptamer train --seqs my_selex.txt --pos 9000 --neg 10000 --out runs/my_target
python -m deepaptamer predict --weights runs/my_target --input candidates.txt -o predictions.tsv
```

Training fits all three models (CNN, BiLSTM, DeepAptamer) for 100 epochs, as the paper did.
It then scores a held-out 10% and saves `metrics.tsv` and `roc_pr.png` next to the weights.
On an Apple-silicon laptop, 10,000 sequences take roughly 20 minutes.
Use `--models cnn_bilstm` to train only DeepAptamer, or `--epochs` to shorten a trial run.

Expect a warning that a cutoff falls inside a group of tied counts. That is normal — most reads
are singletons — but it means which of those sequences get used depends on your file's row
order. For a reproducible run, sort the tied rows yourself before training.

## Reproduction

`python -m deepaptamer evaluate --target CTGF` rebuilds the paper's held-out test set, scores it
with the shipped weights, and writes `results/CTGF/`.

| Target | CNN | BiLSTM | DeepAptamer |
|---|---|---|---|
| CTGF (published) | 0.897 | 0.905 | 0.919 |
| CTGF (this code) | 0.897 | 0.905 | 0.909 |
| BCMA (published) | 0.987 | 0.990 | 0.990 |
| BCMA (this code) | 0.987 | 0.990 | 0.989 |

The values are AUROC. The DKK1 weights in `model/DKK1` do not reproduce the published result on
`data/sequence/DK-30.txt` (AUROC ≈ 0.59), so DKK1 is not offered as a target.

## Things to know

- **The supplied models are barely better than their own baselines.** On CTGF, DeepAptamer leads
  the plain BiLSTM 0.909 to 0.905; on BCMA the BiLSTM wins outright. The paper's margins were
  also obtained by searching the non-binder count until DeepAptamer beat the baselines on the
  *test* split (`tune()` in `script/main.py`), so treat the hybrid's advantage as unproven.
- **Held-out scores are optimistic, badly so for BCMA.** SELEX enriches families of near-identical
  sequences, and a random 90/10 split puts siblings on both sides. The median held-out BCMA binder
  sits 1 mutation from a training sequence (CTGF: 10). Expect worse performance on sequences
  unlike anything in training, and prefer the CTGF-style numbers as the honest guide.
- **A score is "looks enriched", not "binds tightly".** Enrichment is a proxy for affinity that
  also reflects PCR bias and amplification; validate hits experimentally.
- **Inputs are fixed at 35 nt.** Both the architecture and the shape features assume this length.
- **The attention layer mixes batches.** In the original model it mixes each sequence with the
  others in its batch, so a sequence's score depended on its batch-mates (by up to 0.6 in testing).
  This package scores every sequence on its own, which is the same as a batch of one, so a score
  doesn't depend on what else is in the file. On the test sets this changes AUROC by ≤ 0.001.
- **Weights need tf_keras.** The `.h5` files contain weights only, saved with Keras 2.4, so the
  architectures are rebuilt in `deepaptamer/models.py` with `tf_keras` (Keras 2).
- **Shape features are a numpy port of DNAshapeR**, not a call out to R. `tests/test_shape.py`
  checks it against stored DNAshapeR 1.38.0 output, including the `N`-padding edge cases.
  It is about 125× faster than the R round-trip (~185,000 sequences/sec).

## Layout

| Path | Contents |
|---|---|
| `app.py` | The Streamlit web app |
| `deepaptamer/` | Runnable package: `prep.py` (reads → counted sequences), `shape.py` + `shape_tables.npz` (DNA shape in numpy), `features.py`, `models.py`, `metrics.py`, CLI in `__main__.py` |
| `tests/` | `pytest tests/` — shape encoding against DNAshapeR, and read preparation |
| `tools/` | Regenerate the shape tables from R; not needed to use the package |
| `data/sequence/` | SELEX sequences with frequencies, highest first (CT-20 = CTGF, BC-6 = BCMA, DK-30 = DKK1). CT-20 and BC-6 are already trimmed to the authors' binder and non-binder sets, with the middle of the range removed. |
| `model/` | Authors' trained weights and result plots. Folder names give the binder/non-binder counts (`9k_1w` = 9,000 / 10,000; w = 10,000). |
| `script/` | Authors' original interactive scripts, kept for reference. They contain hardcoded paths and won't run as-is. |
