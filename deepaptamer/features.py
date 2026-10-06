"""Turn 35-nt aptamer sequences into DeepAptamer's two inputs.

Each sequence becomes a 35x4 one-hot block and 126 DNA shape values (MGW 31, ProT 31, Roll 32,
HelT 32) — the same layout DNAshapeR's encodeSeqShape(c("1-mer", "1-shape")) produces, which is
what the original training files held. The shape values come from `shape.py`, a numpy port of
DNAshapeR verified to match it exactly, so no R installation is required.
"""
import numpy as np

from .shape import N_SHAPE, SEQ_LEN, one_hot, shape_features


def read_sequences(path):
    """Sequences from FASTA, a plain one-per-line list, or a SELEX table (sequence in column 1)."""
    seqs, fasta_parts = [], None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if fasta_parts:
                    seqs.append("".join(fasta_parts))
                fasta_parts = []
            elif fasta_parts is not None:
                fasta_parts.append(line)
            else:
                seqs.append(line.split("\t")[0])
    if fasta_parts:
        seqs.append("".join(fasta_parts))
    return [s.upper() for s in seqs]


def validate(seqs):
    for i, s in enumerate(seqs):
        if len(s) != SEQ_LEN or set(s) - set("ACGTN"):
            raise ValueError(
                f"Sequence {i + 1} ({s!r}) is not a {SEQ_LEN}-nt A/C/G/T/N sequence; "
                f"the models only accept {SEQ_LEN}-nt inputs (pad short reads with trailing N)."
            )


def encode(seqs, normalize=True):
    """Return (onehot [n, 35, 4], shape [n, 126, 1]) as float32 arrays."""
    validate(seqs)
    if not seqs:
        return (np.zeros((0, SEQ_LEN, 4), np.float32), np.zeros((0, N_SHAPE, 1), np.float32))
    return one_hot(seqs), shape_features(seqs, normalize)[:, :, None]


def encode_chunks(seqs, chunk_size=50000, progress=None):
    """Encode in blocks, yielding (onehot, shape) per block, so peak memory stays flat.

    `progress(done, total)` is called after each block.
    """
    total = len(seqs)
    for start in range(0, total, chunk_size):
        yield encode(seqs[start : start + chunk_size])
        if progress:
            progress(min(start + chunk_size, total), total)
