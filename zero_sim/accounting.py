"""Memory and communication accounting.

Two routes to the same numbers:

  rank construction : build the VirtualGPU objects for a stage and world size and
              add up tensor.nbytes. For world sizes other than the 32-rank run
              the tensors are created on PyTorch's "meta" device (shapes and
              dtypes, no storage), so the same construction code runs for free.
              This is the ownership table (cluster.SHARDED) turned into real
              allocations, so agreeing with the formulas checks that table and the
              allocation code; it is not an independent measurement. The real
              32-rank run (engine.run_stage) then checks that real tensors match.

  formula   : the closed forms derived from Session 12's 16-byte model.

The 30B projection multiplies the per-category fractions a rank holds (exactly 1
or 1/N, read off the rank objects) by 30e9 parameters, i.e. it is the course
formula evaluated through our ownership table.
"""
from __future__ import annotations

from .cluster import CATEGORIES, BYTES, make_cluster

GIB = 2 ** 30
CARD_GIB = 80e9 / GIB            # an "80 GB" card is 74.5 GiB (Session 12, section 2)

# Values printed in the Session 12 lesson (sections 6 and 7). Used only as the thing we check against.
COURSE_BYTES_PER_PARAM_8GPU = {0: 16.00, 1: 5.50, 2: 3.75, 3: 2.00}
COURSE_LADDER_30B_GIB = {
    0: {8: 447.0, 16: 447.0, 32: 447.0, 64: 447.0},
    1: {8: 153.7, 16: 132.7, 32: 122.2, 64: 117.0},
    2: {8: 104.8, 16: 80.3, 32: 68.1, 64: 62.0},
    3: {8: 55.9, 16: 27.9, 32: 14.0, 64: 7.0},
}
COURSE_COMM_P = {0: 2, 1: 2, 2: 2, 3: 3}


def course_bytes_per_param(stage: int, n: int) -> float:
    """Closed forms: DP 16, Z1 4 + 12/N, Z2 2 + 14/N, Z3 16/N."""
    return {0: 16.0, 1: 4 + 12 / n, 2: 2 + 14 / n, 3: 16 / n}[stage]


def measured_rank_bytes(stage: int, n: int, layout, rank: int = 0) -> dict:
    """Bytes per category held by one rank, from real VirtualGPU construction on the meta device."""
    g = make_cluster(stage, n, layout, device="meta")[rank]
    return g.persistent_bytes(by_category=True)


def held_fraction(stage: int, n: int, layout) -> dict:
    """Share of each state category one rank holds (1.0 = full replica, 1/N = shard)."""
    total = sum(u.numel for u in layout)
    b = measured_rank_bytes(stage, n, layout)
    return {c: b[c] / (BYTES[c] * total) for c in CATEGORIES}


def measured_bytes_per_param(stage: int, n: int, layout) -> float:
    total = sum(u.numel for u in layout)
    return sum(measured_rank_bytes(stage, n, layout).values()) / total


def project_gib_per_gpu(stage: int, n: int, layout, n_params: float = 30e9) -> float:
    """Per-GPU training state for a model of n_params, using the shard fractions measured on our model."""
    frac = held_fraction(stage, n, layout)
    return sum(BYTES[c] * frac[c] for c in CATEGORIES) * n_params / GIB


def ring_comm_P(stage: int, n: int) -> float:
    """Exact ring volume sent per rank per step in units of P: each ring phase moves (N-1)/N of a buffer."""
    phases = {0: 2, 1: 2, 2: 2, 3: 3}[stage]
    return phases * (n - 1) / n
