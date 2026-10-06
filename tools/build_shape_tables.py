"""Regenerate deepaptamer/shape_tables.npz from a real DNAshapeR install.

You only need this to re-derive the tables from scratch; the committed .npz is what the package
uses and it needs no R. Requires R with DNAshapeR:

    Rscript -e 'install.packages("BiocManager", repos="https://cloud.r-project.org"); BiocManager::install("DNAshapeR")'
    Rscript tools/extract_shape_tables.R      # writes p3_*.tsv and p3_seqs.txt
    python  tools/build_shape_tables.py       # writes deepaptamer/shape_tables.npz

How the tables were derived: DNAshapeR computes each value from a short window of surrounding
bases, so probing it over sequences that cover every pentamer (MGW, ProT) and every hexamer
(Roll, HelT) recovers the lookup exactly. Roll and HelT additionally need the two terminal
steps, where only one pentamer contributes, so the probe set seats every pentamer at both ends.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

NAMES = ["MGW", "ProT", "Roll", "HelT"]
# DNAshapeR's fixed normalisation bounds, and the average-B-DNA value it substitutes for an
# undefined window (normalised). Both read off DNAshapeR:::normalizeShape.
NORM = {"MGW": (2.85, 6.20), "ProT": (-16.51, -0.03), "Roll": (-8.57, 8.64), "HelT": (30.94, 38.05)}
FILL = {"MGW": 0.663284, "ProT": 0.589684, "Roll": 0.457408, "HelT": 0.476231}
CODE = {c: i for i, c in enumerate("ACGT")}


def kidx(s):
    v = 0
    for c in s:
        v = v * 4 + CODE[c]
    return v


def main(src=Path(".")):
    seqs = [l.strip() for l in (src / "p3_seqs.txt").open()]
    out = {}
    for name in NAMES:
        m = np.genfromtxt(src / f"p3_{name}.tsv", delimiter="\t")
        cols = np.where(~np.isnan(m[0]))[0]
        lo, hi = int(cols.min()), int(cols.max())
        if name in ("MGW", "ProT"):
            t = np.full(4**5, np.nan)
            for i, s in enumerate(seqs):
                for j in range(lo, hi + 1):
                    if not np.isnan(m[i, j]):
                        t[kidx(s[j - 2 : j + 3])] = m[i, j]
            out[f"{name}_main"] = t
        else:
            t, fe, le = np.full(4**6, np.nan), np.full(4**5, np.nan), np.full(4**5, np.nan)
            for i, s in enumerate(seqs):
                for j in range(lo + 1, hi):
                    if not np.isnan(m[i, j]):
                        t[kidx(s[j - 2 : j + 4])] = m[i, j]
                if not np.isnan(m[i, lo]):
                    fe[kidx(s[0:5])] = m[i, lo]
                if not np.isnan(m[i, hi]):
                    le[kidx(s[-5:])] = m[i, hi]
            out[f"{name}_main"], out[f"{name}_first"], out[f"{name}_last"] = t, fe, le

    for k, v in out.items():
        if np.isnan(v).any():
            sys.exit(f"{k} is incomplete — the probe set does not cover every k-mer.")
    out["fill"] = np.array([FILL[n] for n in NAMES])
    out["norm_min"] = np.array([NORM[n][0] for n in NAMES])
    out["norm_max"] = np.array([NORM[n][1] for n in NAMES])
    dest = Path(__file__).resolve().parent.parent / "deepaptamer" / "shape_tables.npz"
    np.savez_compressed(dest, **out)
    print(f"wrote {dest} ({dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("."))
