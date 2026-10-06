"""The numpy shape encoder must stay bit-for-bit faithful to DNAshapeR.

`dnashaper_reference.npz` holds real output from DNAshapeR 1.38.0 for 605 sequences — every
N-containing case from the three datasets plus a sample of plain ones. Regenerate it only
against a real R install; see tools/build_shape_tables.py.
"""
from pathlib import Path

import numpy as np
import pytest

from deepaptamer.features import encode
from deepaptamer.prep import parse_reads, prepare, revcomp, trim_one
from deepaptamer.shape import n_shape_for, one_hot, shape_features

REF = np.load(Path(__file__).with_name("dnashaper_reference.npz"))
SEQS = [str(s) for s in REF["seqs"]]
EXPECTED = REF["expected"]


def test_one_hot_matches_dnashaper():
    assert np.abs(one_hot(SEQS).reshape(len(SEQS), -1) - EXPECTED[:, :140]).max() < 1e-6


def test_shape_matches_dnashaper():
    assert np.abs(shape_features(SEQS) - EXPECTED[:, 140:]).max() < 1e-5


def test_shape_matches_on_n_containing_sequences():
    """The N fallback is the subtle part: a step keeps whichever pentamer is still defined."""
    idx = [i for i, s in enumerate(SEQS) if "N" in s]
    assert idx, "fixture should contain N-padded sequences"
    assert np.abs(shape_features([SEQS[i] for i in idx]) - EXPECTED[idx, 140:]).max() < 1e-5


def test_encode_shapes_and_dtypes():
    onehot, shape = encode(SEQS[:8])
    assert onehot.shape == (8, 35, 4) and shape.shape == (8, 126, 1)
    assert onehot.dtype == np.float32 and shape.dtype == np.float32


def test_encode_is_row_independent():
    """A sequence must encode identically alone and in a batch."""
    alone, _ = encode([SEQS[3]])
    batch, _ = encode(SEQS[:8])
    assert np.array_equal(alone[0], batch[3])


def test_encode_empty():
    onehot, shape = encode([])
    assert onehot.shape == (0, 35, 4) and shape.shape == (0, 126, 1)


@pytest.mark.parametrize("bad", ["ACGT", "ACGTX" + "A" * 30])
def test_encode_rejects_bad_input(bad):
    """Too short for a shape window, or not a nucleotide alphabet."""
    with pytest.raises(ValueError):
        encode([bad])


def test_encode_rejects_mixed_lengths():
    with pytest.raises(ValueError, match="same length"):
        encode(["A" * 35, "A" * 36])


def test_encode_rejects_wrong_length_for_a_model():
    with pytest.raises(ValueError, match="model expects 35"):
        encode(["A" * 45], length=35)


@pytest.mark.parametrize("length", [20, 35, 45, 60])
def test_encode_works_at_other_lengths(length):
    """Only the supplied weights are tied to 35 nt; the encoding is not."""
    seq = ("ACGTGGTCA" * 20)[:length]
    onehot, shape = encode([seq])
    assert onehot.shape == (1, length, 4)
    assert shape.shape == (1, n_shape_for(length), 1)


def test_n_shape_formula():
    assert n_shape_for(35) == 126
    assert n_shape_for(45) == 166
    assert all(n_shape_for(L) == 4 * L - 14 for L in range(6, 80))


# ---- read preparation ----

FIVE, THREE = "GGGAGGACGATGCGG", "TTGCGTATTGGG"
CORE = "AAGTATAGACGTGATGCAGGGAATGCAGACCCAGG"


def test_trim_finds_insert_on_both_strands():
    assert trim_one(FIVE + CORE + THREE, FIVE, THREE) == CORE
    assert trim_one(revcomp(FIVE + CORE + THREE), FIVE, THREE) == CORE


def test_trim_reports_missing_primer():
    assert trim_one("ACGT" * 10, FIVE, THREE) is None


def test_prepare_collapses_and_counts():
    reads = parse_reads("\n".join([FIVE + CORE + THREE] * 3).encode())
    rows, info = prepare(reads, FIVE, THREE)
    assert rows == [(CORE, 3)]
    assert info["stats"]["kept"] == 3 and info["stats"]["unique"] == 1


def test_prepare_counts_what_it_drops():
    reads = parse_reads("\n".join([FIVE + CORE + THREE, "ACGTACGTAC"]).encode())
    _, info = prepare(reads, FIVE, THREE)
    assert info["stats"]["no_primer"] == 1


def test_parse_reads_honours_existing_counts():
    rows, _ = prepare(parse_reads(f"{CORE}\t50000\n".encode()))
    assert rows == [(CORE, 50000)]


def test_parse_reads_fastq_and_fasta_agree():
    fq = f"@r\n{CORE}\n+\n{'I' * len(CORE)}\n".encode()
    fa = f">r\n{CORE}\n".encode()
    assert parse_reads(fq) == parse_reads(fa) == [(CORE, 1)]
