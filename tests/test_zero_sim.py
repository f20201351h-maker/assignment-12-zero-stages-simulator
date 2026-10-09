import math
from pathlib import Path

import pytest
import torch

from zero_sim.accounting import (COURSE_BYTES_PER_PARAM_8GPU, COURSE_LADDER_30B_GIB, CARD_GIB,
                                 course_bytes_per_param, measured_bytes_per_param, project_gib_per_gpu,
                                 ring_comm_P, measured_rank_bytes)
from zero_sim.cluster import CATEGORIES, SHARDED, make_cluster, owned_param_slices, shard_range
from zero_sim.collectives import CommLog, naive_all_reduce, ring_all_gather, ring_all_reduce, ring_reduce_scatter
from zero_sim.engine import OptConfig, make_batches, run_single_device, run_stage
from zero_sim.model import GPTConfig, build_layout, init_flat_params

ROOT = Path(__file__).resolve().parents[1]
FULL = GPTConfig()
TINY = GPTConfig(d_model=32, n_layer=2, n_head=2, block_size=16)


# ---------------------------------------------------------------- collectives
@pytest.mark.parametrize("n", [2, 4, 8, 32])
def test_reduce_scatter_then_all_gather_equals_all_reduce(n):
    g = torch.Generator().manual_seed(n)
    bufs = [torch.randn(n * 37, dtype=torch.float64, generator=g) for _ in range(n)]
    log = CommLog(n)
    shards = ring_reduce_scatter(bufs, log)
    total = torch.stack(bufs).sum(0)
    for r, s in enumerate(shards):                       # rank r holds reduced chunk r
        assert torch.allclose(s, total.chunk(n)[r], atol=1e-12)
    rebuilt = ring_all_gather(shards, log)
    naive = naive_all_reduce(bufs)
    for a, b in zip(rebuilt, naive):
        assert (a - b).abs().max() < 1e-12
    # every rank sends (N-1)/N of the buffer in each phase
    m_bytes = bufs[0].numel() * 8
    for call in log.calls:
        assert call["bytes_sent_per_rank"] == [m_bytes * (n - 1) // n] * n


def test_all_gather_is_exact_copy():
    shards = [torch.randn(5).to(torch.bfloat16) for _ in range(8)]
    full = ring_all_gather(shards)
    for f in full:
        assert torch.equal(f, torch.cat(shards))


def test_ring_all_reduce_matches_naive():
    bufs = [torch.randn(64) for _ in range(8)]
    for a, b in zip(ring_all_reduce(bufs), naive_all_reduce(bufs)):
        assert (a - b).abs().max() < 1e-5


# ---------------------------------------------------------------- partitioning
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64])
def test_shards_cover_every_element_exactly_once(n):
    for u in build_layout(FULL):
        covered = torch.zeros(u.numel, dtype=torch.int32)
        for r in range(n):
            lo, hi = shard_range(u.numel, r, n)
            covered[lo:hi] += 1
        assert torch.all(covered == 1)


def test_owned_param_slices_partition_named_tensors():
    layout, n = build_layout(FULL), 32
    counts = {}
    for r in range(n):
        for row in owned_param_slices(layout, r, n):
            a, b = row["local_range"]
            counts[(row["unit"], row["param"])] = counts.get((row["unit"], row["param"]), 0) + (b - a)
    for u in layout:
        for e in u.entries:
            assert counts[(u.name, e.name)] == e.numel


def test_exactly_32_ranks_with_ids():
    ranks = make_cluster(3, 32, build_layout(FULL), device="meta")
    assert [g.rank for g in ranks] == list(range(32))


# ---------------------------------------------------------------- memory accounting
def test_param_count():
    assert sum(u.numel for u in build_layout(FULL)) == 867_072


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64])
def test_measured_bytes_match_course_formula(stage, n):
    assert measured_bytes_per_param(stage, n, build_layout(FULL)) == pytest.approx(course_bytes_per_param(stage, n), abs=1e-12)


def test_course_8gpu_table_and_30b_ladder():
    layout = build_layout(FULL)
    for st, v in COURSE_BYTES_PER_PARAM_8GPU.items():
        assert measured_bytes_per_param(st, 8, layout) == pytest.approx(v, abs=0.005)
    for st, row in COURSE_LADDER_30B_GIB.items():
        for n, gib in row.items():
            assert abs(project_gib_per_gpu(st, n, layout) - gib) <= 0.051, (st, n)
    fits = {st: project_gib_per_gpu(st, 32, layout) <= CARD_GIB for st in range(4)}
    assert fits == {0: False, 1: False, 2: True, 3: True}


def test_stage_ordering_and_no_full_model_in_z3():
    layout, n = build_layout(FULL), 32
    total = [sum(measured_rank_bytes(st, n, layout).values()) for st in range(4)]
    assert total[0] > total[1] > total[2] > total[3]
    p = sum(u.numel for u in layout)
    assert measured_rank_bytes(3, n, layout)["param16"] == 2 * p // n


def test_ring_volume_formula():
    assert ring_comm_P(0, 32) == pytest.approx(2 * 31 / 32)
    assert ring_comm_P(3, 32) == pytest.approx(3 * 31 / 32)


# ---------------------------------------------------------------- numerical equivalence (tiny model, 8 ranks)
@pytest.fixture(scope="module")
def tiny_runs():
    layout = build_layout(TINY)
    init = init_flat_params(TINY, layout, seed=0)
    batches = make_batches((ROOT / "data" / "corpus.txt").read_bytes(), 2, 8, 2, TINY.block_size)
    opt = OptConfig()
    runs = {st: run_stage(st, TINY, opt, init, batches, world=8, probe_rank=3) for st in range(4)}
    ref = run_single_device(TINY, opt, init, batches)
    return runs, ref, layout


def test_dp_replicas_identical(tiny_runs):
    runs, _, _ = tiny_runs
    assert all(v == 0.0 for v in runs[0]["replica_max_diff"].values())
    for st in (1, 2):
        assert runs[st]["replica_max_diff"]["param16"] == 0.0


def test_reduced_gradients_identical_across_stages(tiny_runs):
    runs, _, _ = tiny_runs
    for st in (1, 2, 3):
        for a, b in zip(runs[0]["first_grad"], runs[st]["first_grad"]):
            assert torch.equal(a, b)


def test_zero_updates_match_dp(tiny_runs):
    runs, _, _ = tiny_runs
    for st in (1, 2, 3):
        d = max(float((a - b).abs().max()) for a, b in zip(runs[0]["final"]["master32"], runs[st]["final"]["master32"]))
        assert d <= 1e-5, (st, d)


def test_dp_gradient_matches_single_device(tiny_runs):
    runs, ref, _ = tiny_runs
    num = sum(float((a - b).pow(2).sum()) for a, b in zip(runs[0]["first_grad"], ref["first_grad"]))
    den = sum(float(b.pow(2).sum()) for b in ref["first_grad"])
    assert math.sqrt(num / den) < 8 * 2 ** -9    # bf16 ring reduction: a few bf16 roundings
    assert abs(sum(runs[0]["steps"][0]["losses"]) / 8 - ref["losses"][0]) < 1e-5


def test_model_flops_identical_across_stages(tiny_runs):
    runs, _, _ = tiny_runs
    f = [runs[st]["steps"][0]["probe"] for st in range(4)]
    assert all(x == f[0] for x in f) and f[0]["flops_fwd"] > 0


def test_optimizer_work_partitioned(tiny_runs):
    runs, _, layout = tiny_runs
    p = sum(u.numel for u in layout)
    assert runs[0]["optimizer_elements_by_rank"] == [p] * 8
    for st in (1, 2, 3):
        assert runs[st]["optimizer_elements_by_rank"] == [p // 8] * 8


def test_communication_volume(tiny_runs):
    runs, _, layout = tiny_runs
    p_bytes = 2 * sum(u.numel for u in layout)
    for st in range(4):
        sent = sum(c["bytes_sent_per_rank"][0] for c in runs[st]["log"].calls if c["step"] == 0 and c["tag"] != "grad_norm")
        assert sent == pytest.approx(ring_comm_P(st, 8) * p_bytes)


def test_z3_peak_temp_is_one_unit(tiny_runs):
    runs, _, layout = tiny_runs
    largest = max(u.numel for u in layout)
    # gathered bf16 params + bf16 gradient bucket of the same unit, during that unit's backward
    assert max(runs[3]["peak_temp_by_rank"]) == 2 * largest * 2
    assert max(runs[3]["steps"][0]["held_params_between_fwd_bwd"]) == 2 * sum(u.numel for u in layout) // 8


# ---------------------------------------------------------------- break it
def test_removing_collectives_breaks_agreement():
    layout = build_layout(TINY)
    init = init_flat_params(TINY, layout, seed=0)
    batches = make_batches((ROOT / "data" / "corpus.txt").read_bytes(), 1, 8, 2, TINY.block_size)
    opt = OptConfig()
    assert run_stage(0, TINY, opt, init, batches, world=8, faults={"skip_grad_sync"})["replica_max_diff"]["param16"] > 0
    assert run_stage(1, TINY, opt, init, batches, world=8, faults={"skip_param_allgather"})["replica_max_diff"]["param16"] > 0
    good = run_stage(3, TINY, opt, init, batches, world=8)["steps"][0]["losses"]
    bad = run_stage(3, TINY, opt, init, batches, world=8, faults={"wrong_gather"})["steps"][0]["losses"]
    assert good != bad


def test_zero3_gathered_weights_are_really_freed(monkeypatch):
    """Every bf16 buffer produced by a ZeRO-3 gather must be garbage by the time the next unit is gathered."""
    import gc
    import weakref
    import zero_sim.engine as eng
    refs = []
    real_gather = eng.ring_all_gather

    def tracking_gather(shards, *a, **k):
        gc.collect()
        alive = [r for r in refs if r() is not None]
        assert not alive, f"{len(alive)} gathered tensors from earlier units still alive"
        out = real_gather(shards, *a, **k)
        if k.get("tag", "").startswith("param_gather"):
            refs.extend(weakref.ref(t) for t in out)
        return out

    monkeypatch.setattr(eng, "ring_all_gather", tracking_gather)
    layout = build_layout(TINY)
    init = init_flat_params(TINY, layout, seed=0)
    batches = make_batches((ROOT / "data" / "corpus.txt").read_bytes(), 1, 8, 2, TINY.block_size)
    run_stage(3, TINY, OptConfig(), init, batches, world=8)
    gc.collect()
    assert all(r() is None for r in refs)


def test_all_reduce_bytes_come_from_the_phase_log():
    log = CommLog(8)
    ring_all_reduce([torch.randn(64) for _ in range(8)], log)
    assert log.calls[0]["bytes_sent_per_rank"] == [2 * 7 * 8 * 4] * 8 and log.calls[0]["p2p_sends"] == 2 * 8 * 7
