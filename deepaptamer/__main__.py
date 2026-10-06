"""DeepAptamer command line.

  python -m deepaptamer predict  --target CTGF --input my_seqs.txt
  python -m deepaptamer train    --seqs my_selex.txt --pos 9000 --neg 10000 --out runs/my_target
  python -m deepaptamer predict  --weights runs/my_target --input my_seqs.txt
  python -m deepaptamer evaluate --target CTGF
"""
import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from . import metrics as metrics_mod
from .features import encode, read_sequences
from .models import MODELS, inputs_for, load, predict

REPO = Path(__file__).resolve().parent.parent

# Categorical slots 1-3 of the validated palette, in fixed order: CNN, BiLSTM, DeepAptamer.
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]

# Shipped weights, with the data and binder/non-binder split they were trained and tested on.
TARGETS = {
    "CTGF": dict(seqs="data/sequence/CT-20.txt", pos=9000, neg=10000, weights="model/CTGF/9k_1w"),
    "BCMA": dict(seqs="data/sequence/BC-6.txt", pos=4000, neg=7000, weights="model/BCMA/4k_7k/best_model"),
}


def read_ranked(path):
    """Sequences ranked by enrichment, highest first. If column 2 holds a count or frequency the
    rows are sorted by it; otherwise the file is assumed to be in ranked order already."""
    rows = [line.rstrip("\n").split("\t") for line in open(path) if line.strip()]
    counts = None
    try:
        # A stable sort keeps the file's existing order among equal counts. The shipped weights
        # were trained on exactly that order, so re-ordering ties would change the published
        # results; warn_about_ties() flags when that order is deciding the labels.
        rows.sort(key=lambda r: -float(r[1]))
        counts = [float(r[1]) for r in rows]
    except (IndexError, ValueError):
        pass
    return [r[0].upper() for r in rows], counts


def warn_about_ties(counts, pos, neg):
    """SELEX counts are mostly ties at the low end. If a cutoff lands inside a tied group, which
    sequences get that label comes down to row order, so the same data in a different order would
    train on a different set."""
    if counts is None:
        return
    for name, boundary in (("--pos", pos - 1), ("--neg", len(counts) - neg)):
        value = counts[boundary]
        tied = sum(1 for c in counts if c == value)
        if tied > 1:
            print(f"warning: the {name} cutoff falls inside {tied} sequences that all have "
                  f"count {value:g}; which of them are used depends on the order of your file.")


def load_split(seqs_path, pos, neg):
    """The paper's setup: top `pos` sequences are binders, bottom `neg` are not, 90/10 split."""
    from sklearn.model_selection import train_test_split

    seqs, counts = read_ranked(seqs_path)
    if pos + neg > len(seqs):
        sys.exit(f"{seqs_path} has {len(seqs)} sequences, fewer than --pos + --neg = {pos + neg}.")
    warn_about_ties(counts, pos, neg)
    print(f"Encoding {pos} binders + {neg} non-binders from {seqs_path} ...")
    onehot, shape = encode(seqs[:pos] + seqs[len(seqs) - neg:])
    labels = np.array([1] * pos + [0] * neg)
    return train_test_split(onehot, shape, labels, test_size=0.1, random_state=42)


def report(y, scores, out_dir, title):
    """Print and save the paper's metrics, plus ROC and precision-recall curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = metrics_mod.compute(y, scores)
    curves = metrics_mod.curves(y, scores)

    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(11, 5))
    for row, colour in zip(rows, SERIES_COLOURS):
        label, c = row["model"], curves[row["model"]]
        roc_ax.plot(c["fpr"], c["tpr"], lw=2, color=colour, label=f"{label} = {row['AUROC']:.3f}")
        pr_ax.plot(c["recall"], c["precision"], lw=2, color=colour, label=f"{label} = {row['AUPRC']:.3f}")
    roc_ax.plot([0, 1], [0, 1], color="#9b9a93", ls="--", lw=1)
    roc_ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{title}: ROC")
    pr_ax.set(xlabel="Recall", ylabel="Precision", title=f"{title}: precision-recall")
    for ax in (roc_ax, pr_ax):
        ax.legend(loc="lower right" if ax is roc_ax else "lower left", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr.png", dpi=150)

    print(f"\n{len(y)} held-out sequences ({int(y.sum())} binders)")
    print("".join(f"{h:>12}" for h in metrics_mod.COLUMNS))
    for r in rows:
        print(f"{r['model']:>12}" + "".join(f"{r[c]:>12.3f}" for c in metrics_mod.COLUMNS[1:]))
    with open(out_dir / "metrics.tsv", "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(metrics_mod.COLUMNS)
        w.writerows([[r["model"]] + [f"{r[c]:.4f}" for c in metrics_mod.COLUMNS[1:]] for r in rows])
    print(f"\nSaved {out_dir / 'metrics.tsv'} and {out_dir / 'roc_pr.png'}")


def weight_files(weights_dir):
    found = {name: weights_dir / name / "model.h5" for name in MODELS}
    found = {name: path for name, path in found.items() if path.exists()}
    if not found:
        sys.exit(f"No weights found: expected {weights_dir}/{{cnn,bilstm,cnn_bilstm}}/model.h5")
    return found


def resolve(args):
    """Fill seqs/pos/neg/weights from --target where they weren't given explicitly."""
    preset = TARGETS.get(getattr(args, "target", None), {})
    for key, value in preset.items():
        if key in vars(args) and getattr(args, key) is None:
            setattr(args, key, REPO / value if key in ("seqs", "weights") else value)
    return args


def cmd_predict(args):
    if args.weights is None:
        sys.exit("Give --target (CTGF or BCMA) or --weights DIR.")
    seqs = read_sequences(args.input)
    print(f"Encoding {len(seqs)} sequences from {args.input} ...")
    onehot, shape = encode(seqs)
    scores = {MODELS[name][0]: predict(load(name, path), onehot, shape)
              for name, path in weight_files(Path(args.weights)).items()}
    out = open(args.output, "w", newline="") if args.output else sys.stdout
    w = csv.writer(out, delimiter="\t", lineterminator="\n")
    w.writerow(["sequence"] + list(scores))
    for i, s in enumerate(seqs):
        w.writerow([s] + [f"{p[i]:.4f}" for p in scores.values()])
    if args.output:
        out.close()
        print(f"Wrote {len(seqs)} predictions to {args.output}")


def cmd_evaluate(args):
    if None in (args.seqs, args.pos, args.neg, args.weights):
        sys.exit("Give --target (CTGF or BCMA), or all of --seqs, --pos, --neg and --weights.")
    _, X_test, _, S_test, _, y_test = load_split(args.seqs, args.pos, args.neg)
    scores = {MODELS[name][0]: predict(load(name, path), X_test, S_test)
              for name, path in weight_files(Path(args.weights)).items()}
    report(y_test, scores, Path(args.out or f"results/{args.target or Path(args.weights).name}"),
           args.target or Path(args.seqs).stem)


def cmd_train(args):
    import tf_keras as keras

    X_train, X_test, S_train, S_test, y_train, y_test = load_split(args.seqs, args.pos, args.neg)
    keras.utils.set_random_seed(args.seed)
    y_onehot = keras.utils.to_categorical(y_train, 2)
    out = Path(args.out)
    scores = {}
    for name in args.models:
        label, build = MODELS[name]
        print(f"\nTraining {label} ({args.epochs} epochs, batch size {args.batch_size}) ...")
        model = build()
        model.fit(inputs_for(model, X_train, S_train), y_onehot, batch_size=args.batch_size,
                  epochs=args.epochs, validation_split=0.1, verbose=2)
        (out / name).mkdir(parents=True, exist_ok=True)
        model.save_weights(out / name / "model.h5")
        scores[label] = predict(load(name, out / name / "model.h5"), X_test, S_test)
    report(y_test, scores, out, Path(args.seqs).stem)
    print(f"Weights saved under {out}/; score new sequences with:\n"
          f"  python -m deepaptamer predict --weights {out} --input <file>")


def main():
    parser = argparse.ArgumentParser(prog="python -m deepaptamer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="score your own 35-nt sequences")
    p.add_argument("--input", required=True, help="FASTA, one sequence per line, or a tab-separated table")
    p.add_argument("--target", choices=TARGETS, help="use the shipped weights for this protein")
    p.add_argument("--weights", help="directory with cnn/, bilstm/, cnn_bilstm/ model.h5 files")
    p.add_argument("--output", "-o", help="TSV to write (default: print to the terminal)")
    p.set_defaults(func=cmd_predict)

    for name, func, help_text in [("evaluate", cmd_evaluate, "re-test weights on a SELEX file's held-out 10%%"),
                                  ("train", cmd_train, "train models on your own SELEX data")]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--seqs", help="SELEX table: sequence<TAB>count or frequency (35-nt sequences)")
        p.add_argument("--pos", type=int, help="number of top-ranked sequences to label as binders")
        p.add_argument("--neg", type=int, help="number of bottom-ranked sequences to label as non-binders")
        p.add_argument("--out", help="output directory")
        p.set_defaults(func=func)
        if name == "evaluate":
            p.add_argument("--target", choices=TARGETS, help="re-run the paper's test for this protein")
            p.add_argument("--weights", help="directory with cnn/, bilstm/, cnn_bilstm/ model.h5 files")
        else:
            p.add_argument("--epochs", type=int, default=100, help="default 100, as in the paper")
            p.add_argument("--batch-size", type=int, default=256, help="default 256, as in the paper")
            p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS),
                           help="which models to train (default: all three)")
            p.add_argument("--seed", type=int, default=42)

    args = resolve(parser.parse_args())
    if args.command == "train" and None in (args.seqs, args.pos, args.neg, args.out):
        parser.error("train needs --seqs, --pos, --neg and --out")
    try:
        args.func(args)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
