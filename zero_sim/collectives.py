"""Ring collectives over a list of per-rank tensors, with every send logged.

`bufs[r]` is the tensor held by rank r. Each ring step, every rank sends one
chunk to its right neighbour (r -> r+1). All sends of a step are computed from
the state before the step, which is what simultaneous sends would see.

Reduction arithmetic: the received chunk and the local chunk are added in fp32
(fp64 for fp64 payloads) and the result is stored back in the payload dtype. For a bf16 payload this
rounds once per hop, which is what a bf16 ring reduce does on real hardware.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch


@dataclass
class CommLog:
    world: int
    # one row per collective call
    calls: list = field(default_factory=list)

    def record(self, op, tag, unit, bytes_sent_per_rank, sends, step=None, phase=None):
        self.calls.append(dict(op=op, tag=tag, unit=unit, step=step, phase=phase,
                               bytes_sent_per_rank=list(bytes_sent_per_rank), p2p_sends=sends))

    def bytes_sent(self, rank=None, tag=None, step=None):
        tot = 0
        for c in self.calls:
            if (tag is None or c["tag"] == tag) and (step is None or c["step"] == step):
                tot += c["bytes_sent_per_rank"][rank] if rank is not None else sum(c["bytes_sent_per_rank"])
        return tot

    def summary(self, step=None):
        """bytes sent by one rank (rank 0; the ring is symmetric) and call counts, per tag."""
        out = defaultdict(lambda: dict(calls=0, p2p_sends=0, bytes_per_rank=0))
        for c in self.calls:
            if step is not None and c["step"] != step:
                continue
            row = out[(c["op"], c["tag"])]
            row["calls"] += 1
            row["p2p_sends"] += c["p2p_sends"]
            row["bytes_per_rank"] += c["bytes_sent_per_rank"][0]
        return dict(out)


def _acc_dtype(dtype):
    return torch.promote_types(dtype, torch.float32)       # bf16 -> fp32, fp32 -> fp32, fp64 -> fp64


def _add(a, b):
    acc = _acc_dtype(a.dtype)
    return (a.to(acc) + b.to(acc)).to(a.dtype)


def ring_reduce_scatter(bufs, log: CommLog | None = None, tag="", unit=None, step=None):
    """Every rank starts with a full buffer; rank r ends with the SUM of chunk r over all ranks."""
    n = len(bufs)
    assert all(b.numel() == bufs[0].numel() for b in bufs) and bufs[0].numel() % n == 0
    acc = [list(b.reshape(-1).clone().chunk(n)) for b in bufs]   # acc[rank][chunk]
    sent = [0] * n
    sends = 0
    for s in range(n - 1):
        msgs = []
        for r in range(n):
            c = (r - s - 1) % n
            msgs.append(((r + 1) % n, c, acc[r][c].clone()))      # snapshot = in-flight message
            sent[r] += acc[r][c].numel() * acc[r][c].element_size()
            sends += 1
        for dst, c, payload in msgs:
            acc[dst][c] = _add(acc[dst][c], payload)
    if log is not None:
        log.record("reduce_scatter", tag, unit, sent, sends, step)
    return [acc[r][r] for r in range(n)]


def ring_all_gather(shards, log: CommLog | None = None, tag="", unit=None, step=None):
    """Rank r starts with shard r; every rank ends with the concatenation of all shards."""
    n = len(shards)
    have = [[None] * n for _ in range(n)]
    for r in range(n):
        have[r][r] = shards[r].reshape(-1).clone()
    sent = [0] * n
    sends = 0
    for s in range(n - 1):
        msgs = []
        for r in range(n):
            c = (r - s) % n
            msgs.append(((r + 1) % n, c, have[r][c]))
            sent[r] += have[r][c].numel() * have[r][c].element_size()
            sends += 1
        for dst, c, payload in msgs:
            assert have[dst][c] is None
            have[dst][c] = payload.clone()
    assert all(x is not None for row in have for x in row)
    if log is not None:
        log.record("all_gather", tag, unit, sent, sends, step)
    return [torch.cat(have[r]) for r in range(n)]


def ring_all_reduce(bufs, log: CommLog | None = None, tag="", unit=None, step=None):
    """A ring all-reduce is literally the two phases above, back to back (Session 12, section 4)."""
    phases = CommLog(len(bufs))
    shards = ring_reduce_scatter(bufs, phases)
    full = ring_all_gather(shards, phases)
    if log is not None:          # one entry, with the bytes actually sent in both phases
        sent = [sum(c["bytes_sent_per_rank"][r] for c in phases.calls) for r in range(len(bufs))]
        log.record("all_reduce", tag, unit, sent, sum(c["p2p_sends"] for c in phases.calls), step)
    return full


def naive_all_reduce(bufs, log: CommLog | None = None, tag="", unit=None, step=None):
    """Independent reference: every rank receives every other rank's full buffer and sums in rank order."""
    n = len(bufs)
    total = torch.zeros_like(bufs[0], dtype=_acc_dtype(bufs[0].dtype))
    for b in bufs:
        total += b.to(total.dtype)
    if log is not None:
        log.record("naive_all_reduce", tag, unit, [(n - 1) * bufs[0].numel() * bufs[0].element_size()] * n,
                   n * (n - 1), step)
    return [total.to(bufs[0].dtype).clone() for _ in range(n)]


def all_reduce_scalar(values, log: CommLog | None = None, tag="", step=None):
    """Sum of one fp32 scalar per rank (used for the global gradient norm)."""
    n = len(values)
    total = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in values]).sum()
    if log is not None:   # logged as the 4 bytes each rank contributes; negligible next to P
        log.record("all_reduce", tag, None, [4] * n, n, step)
    return [total.clone() for _ in range(n)]
