"""Group sequences into families, so a train/test split can keep relatives on one side.

SELEX enriches a winning sequence together with the PCR-error variants around it, so the
enriched end of a pool is a few families of near-identical sequences rather than many
independent hits. Splitting those at random puts near-duplicates in both train and test, and
the held-out score then measures recall of memorised motifs instead of generalisation.

A family here is a single-linkage cluster: two sequences join if they differ at no more than
`max_mutations` positions, and the family is the transitive closure of that.
"""
import numpy as np

from .shape import one_hot


def _packed(seqs):
    """(n, 4L) one-hot, flattened. A dot product of two rows counts matching positions."""
    return one_hot(seqs).reshape(len(seqs), -1)


def _length(seqs):
    return len(seqs[0]) if len(seqs) else 0


def families(seqs, max_mutations=5, block=1024):
    """Family id per sequence, as an int array.

    Compares every pair, in blocks so memory stays bounded. Cost grows with the square of the
    number of sequences: fine into the tens of thousands, slow beyond that.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(seqs)
    if n == 0:
        return np.zeros(0, dtype=int)
    L = _length(seqs)
    A = _packed(seqs)
    rows, cols = [], []
    for s in range(0, n, block):
        # matching positions between this block and everything; N matches nothing, which makes
        # an N-padded pair look further apart — the conservative direction for splitting
        d = L - (A[s : s + block] @ A.T)
        r, c = np.nonzero(d <= max_mutations)
        rows.append(r + s)
        cols.append(c)
    r, c = np.concatenate(rows), np.concatenate(cols)
    g = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def family_split(seqs, test_size=0.1, max_mutations=5, random_state=42):
    """Train/test indices with every family kept wholly on one side."""
    from sklearn.model_selection import GroupShuffleSplit

    groups = families(seqs, max_mutations)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train, test = next(splitter.split(np.arange(len(seqs)), groups=groups))
    return train, test, groups


def nearest_training_distance(seqs, train_idx, test_idx, block=512):
    """For each test sequence, the mutation distance to its closest training sequence."""
    L = _length(seqs)
    A, B = _packed([seqs[i] for i in test_idx]), _packed([seqs[i] for i in train_idx])
    out = np.empty(len(A), dtype=np.int32)
    for s in range(0, len(A), block):
        out[s : s + block] = L - (A[s : s + block] @ B.T).max(axis=1)
    return out


def leakage_report(seqs, train_idx, test_idx, labels=None):
    """Summarise how close the test set sits to training. Returns a dict, or None if empty."""
    if len(test_idx) == 0 or len(train_idx) == 0:
        return None
    d = nearest_training_distance(seqs, train_idx, test_idx)
    sel = (labels[test_idx] == 1) if labels is not None else np.ones(len(d), bool)
    if not sel.any():
        sel = np.ones(len(d), bool)
    focus = d[sel]
    return {
        "median": int(np.median(focus)),
        "within_2": int((focus <= 2).sum()),
        "n": int(sel.sum()),
        "leaky": bool(np.median(focus) <= 2),
    }
