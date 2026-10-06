"""DeepAptamer web app: raw sequencing reads in, aptamer affinity scores out.

Run with:  streamlit run app.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from deepaptamer import metrics as metrics_mod
from deepaptamer.features import encode, encode_chunks
from deepaptamer.models import MODELS, inputs_for, load, predict
from deepaptamer.prep import parse_reads, prepare, to_tsv
from deepaptamer.shape import SEQ_LEN

REPO = Path(__file__).resolve().parent
RUNS = REPO / "runs"
CHUNK = 20000

# Categorical slots 1-3 of the validated palette, assigned to models in fixed order.
SERIES = {"CNN": "#2a78d6", "BiLSTM": "#eb6834", "DeepAptamer": "#1baf7a"}
ONE_HUE = "#2a78d6"
AXIS = "#9b9a93"

TARGETS = {
    "CTGF": REPO / "model/CTGF/9k_1w",
    "BCMA": REPO / "model/BCMA/4k_7k/best_model",
}

st.set_page_config(page_title="DeepAptamer", page_icon="\N{DNA DOUBLE HELIX}", layout="wide")


# ---------------------------------------------------------------- helpers


@st.cache_resource(show_spinner=False)
def get_model(name, weights_path, mtime, length):
    """mtime is in the cache key so retrained weights at the same path reload."""
    return load(name, weights_path, length=length)


def weights_length(weights_dir):
    """The length a weight set was trained at; the supplied models predate meta.json."""
    meta = Path(weights_dir) / "meta.json"
    if meta.exists():
        try:
            return int(json.loads(meta.read_text()).get("length", SEQ_LEN))
        except (ValueError, OSError):
            pass
    return SEQ_LEN


def available_weight_sets():
    sets = {f"{name} (supplied)": path for name, path in TARGETS.items() if path.exists()}
    if RUNS.exists():
        for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
            if any((d / m / "model.h5").exists() for m in MODELS):
                sets[f"{d.name} (you trained)"] = d
    return sets


def models_in(weights_dir):
    return {n: weights_dir / n / "model.h5" for n in MODELS if (weights_dir / n / "model.h5").exists()}


def score_sequences(seqs, weights_dir, status, bar):
    """Encode and score in blocks so memory stays flat on a big library."""
    found = models_in(weights_dir)
    L = weights_length(weights_dir)
    bad = [s for s in seqs if len(s) != L]
    if bad:
        raise ValueError(
            f"This model was trained on {L}-nt sequences, but {len(bad):,} of your "
            f"{len(seqs):,} are a different length (first is {len(bad[0])} nt). Weights only "
            f"work at the length they were trained on — train a model on your own data instead."
        )
    loaded = {MODELS[n][0]: get_model(n, str(p), p.stat().st_mtime, L) for n, p in found.items()}
    scores = {label: [] for label in loaded}
    done = 0
    for onehot, shape in encode_chunks(seqs, CHUNK):
        for label, model in loaded.items():
            scores[label].append(predict(model, onehot, shape))
        done = min(done + CHUNK, len(seqs))
        bar.progress(done / len(seqs))
        status.write(f"Scored {done:,} of {len(seqs):,} sequences")
    return {label: np.concatenate(parts) for label, parts in scores.items()}


def base_chart(df):
    return alt.Chart(df).configure_view(stroke=None).configure_axis(
        domainColor=AXIS, tickColor=AXIS, gridColor="#ebeae5", grid=True, labelColor="#52514e",
        titleColor="#52514e",
    ).configure_legend(labelColor="#0b0b0b", titleColor="#52514e")


def curve_chart(curves, rows, x, y, x_title, y_title, value_key, diagonal):
    """ROC or PR, three series, legend carrying the metric value as a direct label."""
    frames = []
    for row in rows:
        c = curves[row["model"]]
        frames.append(pd.DataFrame({
            x: c[x], y: c[y],
            "Model": f"{row['model']} = {row[value_key]:.3f}",
        }))
    df = pd.concat(frames)
    order = [f"{r['model']} = {r[value_key]:.3f}" for r in rows]
    colours = [SERIES[r["model"]] for r in rows]
    line = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X(f"{x}:Q", title=x_title, scale=alt.Scale(domain=[0, 1])),
        y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Model:N", scale=alt.Scale(domain=order, range=colours),
                        legend=alt.Legend(orient="bottom-right", title=None)),
        tooltip=[alt.Tooltip("Model:N"), alt.Tooltip(f"{x}:Q", format=".3f"),
                 alt.Tooltip(f"{y}:Q", format=".3f")],
    )
    if diagonal:
        ref = alt.Chart(pd.DataFrame({x: [0, 1], y: [0, 1]})).mark_line(
            strokeDash=[4, 4], strokeWidth=1, color=AXIS).encode(x=f"{x}:Q", y=f"{y}:Q")
        line = ref + line
    return line.properties(height=320)


def show_results(rows, curves):
    """Table first - it is the relief for the one sub-3:1 hue - then the curves."""
    table = pd.DataFrame(rows).set_index("model")
    st.dataframe(table.style.format("{:.3f}"), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.caption("ROC")
        st.altair_chart(curve_chart(curves, rows, "fpr", "tpr", "False positive rate",
                                    "True positive rate", "AUROC", True), use_container_width=True)
    with right:
        st.caption("Precision-recall")
        st.altair_chart(curve_chart(curves, rows, "recall", "precision", "Recall", "Precision",
                                    "AUPRC", False), use_container_width=True)


def sequences_from_state(key, label):
    """Let a tab use the prepared reads or an upload of its own."""
    have_prep = "rows" in st.session_state
    options = (["Use the reads I prepared in step 1", "Upload a file"] if have_prep
               else ["Upload a file"])
    choice = st.radio(label, options, key=f"{key}_src", horizontal=True)
    if choice.startswith("Use"):
        return st.session_state["rows"]
    upload = st.file_uploader("Sequence file", type=["txt", "tsv", "fa", "fasta", "fastq", "gz"],
                              key=f"{key}_up")
    if upload is None:
        return None
    rows, info = prepare(parse_reads(upload.getvalue()))
    if info["stats"].get("wrong_length") or info["stats"].get("no_primer"):
        st.warning(f"{info['stats'].get('wrong_length', 0):,} reads were not 35 nt and were "
                   "skipped. Trim them in step 1 if that was not expected.")
    return rows


# ---------------------------------------------------------------- layout

st.title("DeepAptamer")
st.caption("Predict which aptamer sequences bind a target protein, from SELEX sequencing reads.")

tab_prep, tab_score, tab_train = st.tabs(
    ["1 · Prepare reads", "2 · Score sequences", "3 · Train a new target"])


# ---- 1. Prepare -------------------------------------------------------

with tab_prep:
    st.subheader("Turn raw reads into counted 35-nt sequences")
    st.write("Upload FASTQ, FASTA or a plain sequence list (`.gz` is fine). Primers are trimmed "
             "off, reads are checked for length, and duplicates are collapsed into counts.")

    upload = st.file_uploader("Sequencing reads",
                              type=["fastq", "fq", "fasta", "fa", "txt", "tsv", "gz"])
    c1, c2 = st.columns(2)
    five = c1.text_input("5' primer (optional)", help="Everything up to and including this is cut.")
    three = c2.text_input("3' primer (optional)", help="Everything from this onward is cut.")
    c3, c4 = st.columns(2)
    pad = c3.checkbox("Pad short inserts with N", value=False,
                      help="Off by default: a padded read carries less information.")
    min_count = c4.number_input("Discard sequences seen fewer than N times", 1, 100, 1)

    if upload is not None and st.button("Prepare reads", type="primary"):
        with st.spinner("Trimming and counting ..."):
            reads = parse_reads(upload.getvalue())
            rows, info = prepare(reads, five, three, pad_short=pad, min_count=int(min_count))
        st.session_state["rows"] = rows
        st.session_state["info"] = info

    if "info" in st.session_state:
        info, rows = st.session_state["info"], st.session_state["rows"]
        s = info["stats"]
        st.divider()
        m = st.columns(4)
        m[0].metric("Reads in", f"{s.get('reads', 0):,}")
        m[1].metric("Usable", f"{s.get('kept', 0):,}")
        m[2].metric("Unique sequences", f"{s.get('unique', 0):,}")
        m[3].metric("Seen only once", f"{s.get('singletons', 0):,}")

        losses = {"No primer match": s.get("no_primer", 0), "Wrong length": s.get("wrong_length", 0),
                  "Ambiguous bases": s.get("bad_bases", 0)}
        dropped = {k: v for k, v in losses.items() if v}
        if dropped:
            st.warning("Dropped — " + ", ".join(f"{k}: {v:,}" for k, v in dropped.items()))
        if not rows:
            st.error("Nothing survived. Check the primer sequences — if the reads are on the "
                     "reverse strand the app already tries the reverse complement, so an empty "
                     "result usually means the primers are wrong.")
        else:
            lengths = pd.DataFrame({"Insert length (nt)": list(info["lengths"]),
                                    "Reads": list(info["lengths"].values())})
            st.caption("Insert length after trimming — the models need exactly 35 nt")
            st.altair_chart(
                alt.Chart(lengths).mark_bar(color=ONE_HUE, cornerRadiusTopLeft=4,
                                            cornerRadiusTopRight=4).encode(
                    # ordinal, not quantitative: a length is a whole number of bases, and a
                    # continuous axis would label ticks at 34.5 nt and squeeze the bars to hairlines
                    x=alt.X("Insert length (nt):O", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Reads:Q"),
                    tooltip=["Insert length (nt)", alt.Tooltip("Reads:Q", format=",")],
                ).properties(height=220), use_container_width=True)

            st.dataframe(pd.DataFrame(rows[:20], columns=["sequence", "count"]),
                         use_container_width=True, hide_index=True)
            st.download_button("Download prepared table (TSV)", to_tsv(rows),
                               file_name="prepared.tsv")
            st.success("Ready. Go to **Score sequences**, or **Train a new target** if this is "
                       "a full SELEX round for a protein with no supplied model.")


# ---- 2. Score ---------------------------------------------------------

with tab_score:
    st.subheader("Score sequences against a trained model")
    sets = available_weight_sets()
    if not sets:
        st.error("No weights found.")
    else:
        choice = st.selectbox("Model", list(sets))
        weights_dir = sets[choice]
        if choice.endswith("(supplied)"):
            st.caption("Supplied models only cover CTGF and BCMA. For any other protein, train "
                       "one in step 3.")
        rows = sequences_from_state("score", "Sequences to score")

        if rows and st.button("Score", type="primary", key="score_go"):
            seqs = [s for s, _ in rows]
            bar, status = st.progress(0.0), st.empty()
            try:
                scores = score_sequences(seqs, weights_dir, status, bar)
            except ValueError as e:
                bar.empty(); status.empty()
                st.error(str(e))
                st.stop()
            status.empty()
            out = pd.DataFrame({"sequence": seqs, "count": [c for _, c in rows], **scores})
            headline = "DeepAptamer" if "DeepAptamer" in scores else list(scores)[0]
            st.session_state["scored"] = (out, headline)

        if "scored" in st.session_state:
            out, headline = st.session_state["scored"]
            st.divider()
            hits = int((out[headline] > 0.5).sum())
            c = st.columns(3)
            c[0].metric("Sequences scored", f"{len(out):,}")
            c[1].metric(f"Predicted binders ({headline})", f"{hits:,}")
            c[2].metric("Highest score", f"{out[headline].max():.3f}")

            st.caption(f"Distribution of {headline} scores")
            st.altair_chart(
                alt.Chart(out).mark_bar(color=ONE_HUE, cornerRadiusTopLeft=4,
                                        cornerRadiusTopRight=4).encode(
                    x=alt.X(f"{headline}:Q", bin=alt.Bin(maxbins=40), title="Predicted probability"),
                    y=alt.Y("count()", title="Sequences"),
                    tooltip=[alt.Tooltip("count()", title="Sequences", format=",")],
                ).properties(height=240), use_container_width=True)

            st.dataframe(out.sort_values(headline, ascending=False).head(100),
                         use_container_width=True, hide_index=True)
            st.download_button("Download all scores (TSV)", out.to_csv(sep="\t", index=False),
                               file_name="scores.tsv")


# ---- 3. Train ---------------------------------------------------------

with tab_train:
    st.subheader("Train a model for your own target")
    st.write("Needs one full SELEX round: the most enriched sequences become binders and the "
             "least enriched become non-binders, so a file of only your good hits will not work.")

    rows = sequences_from_state("train", "SELEX reads")
    if rows:
        n = len(rows)
        st.caption(f"{n:,} unique sequences available.")
        c1, c2 = st.columns(2)
        pos = c1.number_input("Binders (most enriched)", 100, max(100, n // 2),
                              min(9000, n // 3), step=500)
        neg = c2.number_input("Non-binders (least enriched)", 100, max(100, n // 2),
                              min(10000, n // 3), step=500)
        c3, c4 = st.columns(2)
        name = c3.text_input("Name for this model", "my_target")
        epochs = c4.number_input("Epochs", 1, 300, 100, step=10)
        chosen = st.multiselect("Models to train", list(MODELS),
                                default=list(MODELS),
                                format_func=lambda k: MODELS[k][0])
        split_mode = st.radio(
            "How to hold out the test set", ["Family-aware (recommended)", "Random"],
            horizontal=True, key="split_mode",
            help="SELEX enriches families of near-identical sequences. A random split puts "
                 "relatives in both train and test, so the score flatters the model. "
                 "Family-aware keeps each family wholly on one side.")

        if int(pos) + int(neg) > n:
            st.error(f"Binders + non-binders ({int(pos) + int(neg):,}) exceeds the "
                     f"{n:,} sequences available.")
        elif st.button("Start training", type="primary", disabled=not chosen):
            import tf_keras as keras
            from sklearn.model_selection import train_test_split

            counts = [c for _, c in rows]
            if counts[int(pos) - 1] == counts[-int(neg)]:
                st.warning("The binder and non-binder cutoffs have the same read count, so the "
                           "two classes are not actually separated. Lower one of them.")
            seqs = [s for s, _ in rows[: int(pos)]] + [s for s, _ in rows[-int(neg):]]
            labels = np.array([1] * int(pos) + [0] * int(neg))

            bar, status = st.progress(0.0), st.empty()
            status.write("Encoding sequences ...")
            try:
                blocks = list(encode_chunks(seqs, CHUNK,
                                            progress=lambda d, t: bar.progress(d / t)))
            except ValueError as e:
                st.error(str(e))
                st.stop()
            onehot = np.concatenate([b[0] for b in blocks])
            shape = np.concatenate([b[1] for b in blocks])
            L = onehot.shape[1]

            if split_mode.startswith("Family"):
                from deepaptamer.cluster import family_split

                status.write("Grouping sequences into families ...")
                tr, te, groups = family_split(seqs)
                st.caption(f"{len(set(groups)):,} families among {len(seqs):,} sequences; "
                           f"{len(tr):,} train / {len(te):,} test, no family spanning both.")
            else:
                tr, te = train_test_split(np.arange(len(seqs)), test_size=0.1, random_state=42)

            from deepaptamer.cluster import leakage_report

            rep = leakage_report(seqs, tr, te, labels)
            if rep and rep["leaky"]:
                st.warning(
                    f"Held-out binders sit a median of {rep['median']} of {L} mutations from the "
                    f"nearest training sequence ({rep['within_2']} of {rep['n']} within 2). These "
                    f"scores largely measure recall of memorised sequences. Use a family-aware "
                    f"split for a number that reflects novel sequences.")

            Xo_tr, Xo_te, Xs_tr, Xs_te = onehot[tr], onehot[te], shape[tr], shape[te]
            y_tr, y_te = labels[tr], labels[te]
            y_onehot = keras.utils.to_categorical(y_tr, 2)

            out_dir = RUNS / name
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "meta.json").write_text(json.dumps(
                {"length": int(L), "split": "family" if split_mode.startswith("Family") else "random"},
                indent=2))
            scores = {}

            class Progress(keras.callbacks.Callback):
                def __init__(self, label, total):
                    super().__init__()
                    self.label, self.total = label, total

                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    bar.progress((epoch + 1) / self.total)
                    status.write(f"{self.label} — epoch {epoch + 1}/{self.total} · "
                                 f"loss {logs.get('loss', 0):.4f} · "
                                 f"val_loss {logs.get('val_loss', 0):.4f}")

            keras.utils.set_random_seed(42)
            for key in chosen:
                label, build = MODELS[key]
                bar.progress(0.0)
                model = build(length=L)
                model.fit(inputs_for(model, Xo_tr, Xs_tr), y_onehot, batch_size=256,
                          epochs=int(epochs), validation_split=0.1, verbose=0,
                          callbacks=[Progress(label, int(epochs))])
                (out_dir / key).mkdir(parents=True, exist_ok=True)
                model.save_weights(out_dir / key / "model.h5")
                scores[label] = predict(
                    load(key, out_dir / key / "model.h5", length=L), Xo_te, Xs_te)

            status.empty()
            bar.empty()
            st.session_state["trained"] = (metrics_mod.compute(y_te, scores),
                                           metrics_mod.curves(y_te, scores), name, len(y_te))

    if "trained" in st.session_state:
        rows_m, curves, name, n_test = st.session_state["trained"]
        st.divider()
        st.write(f"**Held-out performance** — {n_test:,} sequences the models never saw")
        show_results(rows_m, curves)
        st.success(f"Saved to `runs/{name}/`. It is now in the model list on the "
                   "**Score sequences** tab.")
