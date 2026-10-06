"""DNA shape features in pure numpy, reproducing DNAshapeR's `1-shape` encoding.

DNAshapeR derives each shape value from a short window of surrounding bases, so the whole
calculation is a table lookup. `shape_tables.npz` holds those tables, extracted from DNAshapeR
1.38.0 by querying it over sequences covering every pentamer and hexamer; `tools/build_shape_tables.py`
regenerates them. Output matches DNAshapeR exactly (see `tests/test_shape.py`), which removes the
R dependency and makes encoding a few hundred times faster.

Layout, for a 35-nt sequence, matching encodeSeqShape(c("1-shape")):
    MGW  31 values  position j in 2..32, from the pentamer s[j-2:j+3]
    ProT 31 values  likewise
    Roll 32 values  j in 1..32, from the hexamer s[j-2:j+4]; the two end steps fall back
    HelT 32 values  to the pentamer at that end, which has no neighbour to average with
A window containing N has no defined value, so it takes that shape's average-B-DNA constant,
which is what DNAshapeR substitutes.
"""
from pathlib import Path

import numpy as np

SEQ_LEN = 35
N_SHAPE = 126
NAMES = ["MGW", "ProT", "Roll", "HelT"]
_BASES = "ACGT"

_T = np.load(Path(__file__).with_name("shape_tables.npz"))
_FILL = _T["fill"]
_MIN, _MAX = _T["norm_min"], _T["norm_max"]

_LOOKUP = np.full(256, -1, dtype=np.int8)
for _i, _c in enumerate(_BASES):
    _LOOKUP[ord(_c)] = _i
    _LOOKUP[ord(_c.lower())] = _i


def _codes(seqs):
    """(n, L) base codes, with -1 for N or anything unrecognised."""
    raw = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8)
    return _LOOKUP[raw].reshape(len(seqs), -1)


def _kmer_index(codes, start, k):
    """Index of the k-mer beginning at `start`, and a mask of which rows are valid."""
    window = codes[:, start : start + k]
    valid = (window >= 0).all(axis=1)
    idx = np.zeros(len(codes), dtype=np.int64)
    for t in range(k):
        idx = idx * 4 + np.maximum(window[:, t], 0)
    return idx, valid


def shape_features(seqs, normalize=True):
    """(n, 126) float32 shape features, in DNAshapeR's MGW/ProT/Roll/HelT column order."""
    codes = _codes(seqs)
    if codes.shape[1] != SEQ_LEN:
        raise ValueError(f"shape features need {SEQ_LEN}-nt sequences, got {codes.shape[1]}")
    n = len(seqs)
    blocks = []

    for s, name in enumerate(NAMES):
        if name in ("MGW", "ProT"):
            table = _T[f"{name}_main"]
            cols = range(2, SEQ_LEN - 2)                      # 31 positions
            out = np.empty((n, len(cols)), dtype=np.float64)
            for c, j in enumerate(cols):
                idx, ok = _kmer_index(codes, j - 2, 5)
                out[:, c] = np.where(ok, table[idx], np.nan)
        else:
            # A step sits between two overlapping pentamers and takes their average. Where one of
            # them spans an N, DNAshapeR keeps the other rather than discarding the step, so the
            # single-pentamer tables are the fallback, not the fill.
            main, first, last = (_T[f"{name}_main"], _T[f"{name}_first"], _T[f"{name}_last"])
            cols = range(1, SEQ_LEN - 2)                      # 32 steps
            out = np.empty((n, len(cols)), dtype=np.float64)
            for c, j in enumerate(cols):
                left_ok = right_ok = None
                if j > 1:
                    li, left_ok = _kmer_index(codes, j - 2, 5)
                if j < SEQ_LEN - 3:
                    ri, right_ok = _kmer_index(codes, j - 1, 5)

                if left_ok is None:                            # first step: right pentamer only
                    out[:, c] = np.where(right_ok, first[ri], np.nan)
                elif right_ok is None:                         # last step: left pentamer only
                    out[:, c] = np.where(left_ok, last[li], np.nan)
                else:
                    hi_, both = _kmer_index(codes, j - 2, 6)
                    vals = np.where(both, main[hi_],
                                    np.where(left_ok, last[li],
                                             np.where(right_ok, first[ri], np.nan)))
                    out[:, c] = vals

        if normalize:
            out = (out - _MIN[s]) / (_MAX[s] - _MIN[s])
            out = np.where(np.isnan(out), _FILL[s], out)
        blocks.append(out)

    return np.concatenate(blocks, axis=1).astype(np.float32)


def one_hot(seqs):
    """(n, 35, 4) one-hot, with an all-zero column for N — the same as DNAshapeR's 1-mer block."""
    codes = _codes(seqs)
    out = np.zeros((len(seqs), SEQ_LEN, 4), dtype=np.float32)
    rows, cols = np.nonzero(codes >= 0)
    out[rows, cols, codes[rows, cols]] = 1.0
    return out
