"""Turn raw SELEX sequencing reads into the sequence+count table the models use.

A read off the sequencer looks like [5' primer][randomised region][3' primer]. The models only
take the randomised region, at exactly 35 nt, so the primers come off first and the surviving
inserts are collapsed into counts.
"""
import gzip
from collections import Counter

SEQ_LEN = 35
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def revcomp(seq):
    return seq.translate(COMPLEMENT)[::-1]


def _decode(raw):
    if raw[:2] == b"\x1f\x8b":  # gzip magic number
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def parse_reads(raw):
    """Read sequences out of FASTQ, FASTA, or one-per-line text (optionally gzipped).

    Returns a list of (sequence, weight) pairs. Raw reads weigh 1 each; a tab-separated file is
    read as sequence in column 1 and, when column 2 is a number, that already-collapsed count
    becomes the weight.
    """
    text = _decode(raw) if isinstance(raw, (bytes, bytearray)) else raw
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return []

    if lines[0].startswith("@"):  # FASTQ: 4 lines per record
        return [(lines[i].upper(), 1) for i in range(1, len(lines), 4)]

    if lines[0].startswith(">"):  # FASTA, possibly wrapped over several lines
        seqs, current = [], []
        for ln in lines:
            if ln.startswith(">"):
                if current:
                    seqs.append(("".join(current).upper(), 1))
                current = []
            else:
                current.append(ln)
        if current:
            seqs.append(("".join(current).upper(), 1))
        return seqs

    reads = []
    for ln in lines:  # plain list, or an already-collapsed sequence+count table
        parts = ln.split("\t")
        weight = 1
        if len(parts) > 1:
            try:
                weight = max(1, int(float(parts[1])))
            except ValueError:
                weight = 1
        reads.append((parts[0].upper(), weight))
    return reads


def trim_one(seq, five="", three="", search_revcomp=True):
    """Cut the insert out of one read. Returns None when a given primer isn't present."""
    for candidate in (seq, revcomp(seq)) if search_revcomp else (seq,):
        work = candidate
        if five:
            i = work.find(five)
            if i < 0:
                continue
            work = work[i + len(five):]
        if three:
            j = work.find(three)
            if j < 0:
                continue
            work = work[:j]
        return work
    return None


def prepare(reads, five="", three="", pad_short=False, min_count=1):
    """Trim, length-filter and collapse reads.

    Returns (rows, stats) where rows is a list of (sequence, count) sorted by count descending.
    """
    five, three = five.strip().upper(), three.strip().upper()
    stats = Counter()
    lengths = Counter()
    counts = Counter()

    for read, weight in reads:
        stats["reads"] += weight
        insert = trim_one(read, five, three) if (five or three) else read
        if insert is None:
            stats["no_primer"] += weight
            continue
        lengths[len(insert)] += weight
        if len(insert) < SEQ_LEN and pad_short:
            insert = insert + "N" * (SEQ_LEN - len(insert))
            stats["padded"] += weight
        if len(insert) != SEQ_LEN:
            stats["wrong_length"] += weight
            continue
        if set(insert) - set("ACGTN"):
            stats["bad_bases"] += weight
            continue
        counts[insert] += weight

    rows = [(s, c) for s, c in counts.most_common() if c >= min_count]
    stats["kept"] = sum(c for _, c in rows)
    stats["unique"] = len(rows)
    stats["singletons"] = sum(1 for _, c in rows if c == 1)
    return rows, {"stats": dict(stats), "lengths": dict(sorted(lengths.items()))}


def to_tsv(rows):
    return "".join(f"{s}\t{c}\n" for s, c in rows)
