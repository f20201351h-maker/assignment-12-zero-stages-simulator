"""32 virtual GPUs: explicit ranks that own real tensors.

A VirtualGPU is a plain Python object with a rank id and a dict of tensors
(`persistent`) that survive between training steps. Which tensors it holds,
and whether each one is the full unit or a 1/N shard, is decided only by the
ZeRO stage. Temporary buffers (gathered parameters, gradient buckets) are
tracked separately with a running peak.

Course accounting (Session 12, section 1), one parameter:
    param16   2 B   16-bit weight used by forward/backward
    grad16    2 B   16-bit gradient
    master32  4 B   fp32 master copy of the weight
    adam_m    4 B   Adam first moment
    adam_v    4 B   Adam second moment
The tensors below use exactly these dtypes, so tensor.nbytes IS the course accounting.
"""
from __future__ import annotations

import torch

CATEGORIES = ("param16", "grad16", "master32", "adam_m", "adam_v")
DTYPES = {"param16": torch.bfloat16, "grad16": torch.bfloat16,
          "master32": torch.float32, "adam_m": torch.float32, "adam_v": torch.float32}
BYTES = {c: torch.tensor([], dtype=d).element_size() for c, d in DTYPES.items()}

# Which categories are partitioned (1/N per rank) at each stage. Everything else is a full replica.
SHARDED = {
    0: frozenset(),
    1: frozenset({"master32", "adam_m", "adam_v"}),
    2: frozenset({"grad16", "master32", "adam_m", "adam_v"}),
    3: frozenset({"param16", "grad16", "master32", "adam_m", "adam_v"}),
}
STAGE_NAMES = {0: "DP (ZeRO-0)", 1: "ZeRO-1", 2: "ZeRO-2", 3: "ZeRO-3"}


def shard_range(numel: int, rank: int, world: int) -> tuple[int, int]:
    assert numel % world == 0, "units are sized to divide evenly; no padding needed"
    s = numel // world
    return rank * s, (rank + 1) * s


class VirtualGPU:
    def __init__(self, rank: int, world: int, stage: int, layout, device="cpu"):
        self.rank, self.world, self.stage, self.layout = rank, world, stage, layout
        self.persistent: dict[tuple[str, int], torch.Tensor] = {}
        for cat in CATEGORIES:
            for u in layout:
                n = u.numel // world if cat in SHARDED[stage] else u.numel
                self.persistent[(cat, u.index)] = torch.zeros(n, dtype=DTYPES[cat], device=device)
        self.temp: dict[str, torch.Tensor] = {}
        self.temp_bytes = 0
        self.peak_temp_bytes = 0
        self.temp_trace: list[tuple[str, int]] = []    # (event, live temp bytes) for plotting
        self.optimizer_elements = 0                    # elements this rank ran Adam on (cumulative)
        self.batch = None                              # (inputs, targets) for the current step

    # ---------- ownership ----------
    def is_sharded(self, cat):
        return cat in SHARDED[self.stage]

    def owned_range(self, unit_index):
        return shard_range(self.layout[unit_index].numel, self.rank, self.world)

    def get(self, cat, unit_index):
        return self.persistent[(cat, unit_index)]

    # ---------- memory ----------
    def persistent_bytes(self, by_category=False):
        per = {c: 0 for c in CATEGORIES}
        for (cat, _), t in self.persistent.items():
            per[cat] += t.nbytes
        return per if by_category else sum(per.values())

    def hold_temp(self, key, tensor, event=""):
        assert key not in self.temp, key
        self.temp[key] = tensor
        self.temp_bytes += tensor.nbytes
        self.peak_temp_bytes = max(self.peak_temp_bytes, self.temp_bytes)
        self.temp_trace.append((f"+{event or key}", self.temp_bytes))

    def release_temp(self, key, event=""):
        t = self.temp.pop(key)
        self.temp_bytes -= t.nbytes
        self.temp_trace.append((f"-{event or key}", self.temp_bytes))

    def describe(self):
        """Human-readable list of what this rank keeps between steps."""
        rows = []
        for cat in CATEGORIES:
            held = sum(self.get(cat, u.index).numel() for u in self.layout)
            total = sum(u.numel for u in self.layout)
            rows.append(dict(category=cat, holds="1/%d shard" % self.world if self.is_sharded(cat) else "full",
                             elements=held, of_total=total, bytes=held * BYTES[cat]))
        return rows


def make_cluster(stage, world, layout, device="cpu"):
    ranks = [VirtualGPU(r, world, stage, layout, device) for r in range(world)]
    assert len(ranks) == world and [g.rank for g in ranks] == list(range(world))
    return ranks


def owned_param_slices(layout, rank, world):
    """For each unit, which named tensors (and which flat element range inside each) rank owns."""
    out = []
    for u in layout:
        lo, hi = shard_range(u.numel, rank, world)
        for e in u.entries:
            a, b = max(lo, e.offset), min(hi, e.offset + e.numel)
            if a < b:
                start, stop = a - e.offset, b - e.offset
                desc = f"elements [{start}, {stop}) of {e.numel}"
                if len(e.shape) == 2:
                    cols = e.shape[1]
                    desc += f" = rows {start // cols}..{(stop - 1) // cols} (row-major, {cols} cols)"
                out.append(dict(unit=u.name, unit_flat_range=(lo, hi), param=e.name,
                                shape=list(e.shape), local_range=(start, stop), detail=desc))
    return out
