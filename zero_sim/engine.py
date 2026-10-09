"""One training step for DP / ZeRO-1 / ZeRO-2 / ZeRO-3 on the virtual cluster.

All ranks advance in lockstep, one unit at a time, the way SPMD code runs on a
real cluster: every rank reaches the same collective together.

The four stages share one code path. Only two things change with the stage:

  where forward/backward get their parameters from
      DP, Z1, Z2 : the rank's own full bf16 copy
      Z3         : an all-gather of the unit, released right after use, and
                   gathered again for the backward pass

  where the gradients go after a unit's backward
      DP : all-reduce into the rank's full gradient buffer
      Z1 : reduce-scatter; the owner writes the averaged shard into its region
           of its (still full) gradient buffer
      Z2, Z3 : reduce-scatter of a temporary bucket; the owner keeps only its
           shard, the bucket is freed

The optimizer then runs on the full tensor (DP) or on the owned shard only
(Z1-Z3), and Z1/Z2 all-gather the updated bf16 shards so every replica agrees.

Forward/backward arithmetic runs in fp32 on an upcast "compute copy" of the
bf16 weights. That copy is a CPU convenience and is not counted as model state.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict

import torch
from torch.utils.flop_counter import FlopCounterMode

from .cluster import make_cluster, STAGE_NAMES
from .collectives import CommLog, ring_all_gather, ring_all_reduce, ring_reduce_scatter, all_reduce_scalar
from .model import GPTConfig, build_layout, unit_forward, lm_loss


@dataclass(frozen=True)
class OptConfig:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    clip: float = 1.0          # global-norm gradient clipping, as in the V4 DeepSpeed config

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- data
def make_batches(corpus: bytes, steps, world, micro_bs, block, seed=1):
    data = torch.tensor(list(corpus), dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(steps):
        starts = torch.randint(0, len(data) - block - 1, (world * micro_bs,), generator=g)
        x = torch.stack([data[s:s + block] for s in starts])
        y = torch.stack([data[s + 1:s + block + 1] for s in starts])
        out.append((x, y))
    return out


def split_batch(x, y, world):
    return list(zip(x.chunk(world), y.chunk(world)))


# ---------------------------------------------------------------- init
def load_initial_state(ranks, init_flats):
    for g in ranks:
        for u in g.layout:
            full32 = init_flats[u.index]
            lo, hi = g.owned_range(u.index)
            for cat, src in (("master32", full32), ("param16", full32.to(torch.bfloat16))):
                g.get(cat, u.index).copy_(src[lo:hi] if g.is_sharded(cat) else src)


# ---------------------------------------------------------------- parameter fetch / release
def fetch_unit_params(ranks, u, log, step, phase, faults):
    """Full bf16 parameters of unit u for every rank."""
    if ranks[0].stage < 3:
        return [g.get("param16", u.index) for g in ranks]
    shards = [g.get("param16", u.index) for g in ranks]
    if "wrong_gather" in faults:      # deliberately feed the wrong owner's shard (break-it demo)
        shards = shards[1:] + shards[:1]
    full = ring_all_gather(shards, log, tag=f"param_gather_{phase}", unit=u.name, step=step)
    for g, t in zip(ranks, full):
        g.hold_temp(f"gathered:{u.name}", t, event=f"gather {u.name} ({phase})")
    return full


def release_unit_params(ranks, u, phase):
    if ranks[0].stage == 3:
        for g in ranks:
            g.release_temp(f"gathered:{u.name}", event=f"release {u.name} ({phase})")


# ---------------------------------------------------------------- the step
def train_step(ranks, cfg: GPTConfig, opt: OptConfig, step: int, log: CommLog,
               faults=frozenset(), probe_rank=None):
    N, stage, layout = len(ranks), ranks[0].stage, ranks[0].layout
    U = len(layout)
    probe = dict(flops_fwd=0, flops_bwd=0)

    xin = [[None] * U for _ in range(N)]
    outs = [[None] * U for _ in range(N)]
    leaves = [[None] * U for _ in range(N)]
    refetched = {}                       # (rank, unit) -> fp32 compute copy used during backward
    act_seen = [set() for _ in range(N)]
    act_bytes = [0] * N

    def hooks(r, ui, storage_ptr):
        def pack(t):
            if t.untyped_storage().data_ptr() == storage_ptr:          # a weight, not an activation
                return ("W", t.size(), t.stride(), t.storage_offset())
            p = t.untyped_storage().data_ptr()
            if p not in act_seen[r]:
                act_seen[r].add(p)
                act_bytes[r] += t.untyped_storage().nbytes()
            return t

        def unpack(obj):
            if isinstance(obj, tuple) and obj and obj[0] == "W":
                return refetched[(r, ui)].as_strided(obj[1], obj[2], obj[3])
            return obj
        return pack, unpack

    # ---------------- forward, unit by unit
    h = [g.batch[0] for g in ranks]
    for u in layout:
        full16 = fetch_unit_params(ranks, u, log, step, "fwd", faults)
        for r, g in enumerate(ranks):
            leaf = full16[r].float().requires_grad_()
            x_in = h[r] if u.kind == "embed" else h[r].detach().requires_grad_()
            pack, unpack = hooks(r, u.index, leaf.untyped_storage().data_ptr())
            fc = FlopCounterMode(display=False) if r == probe_rank else None
            with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
                if fc:
                    with fc:
                        out = unit_forward(u, cfg, x_in, u.views(leaf))
                        if u.kind == "head":
                            out = lm_loss(out, g.batch[1])
                    probe["flops_fwd"] += fc.get_total_flops()
                else:
                    out = unit_forward(u, cfg, x_in, u.views(leaf))
                    if u.kind == "head":
                        out = lm_loss(out, g.batch[1])
            leaf.untyped_storage().resize_(0)          # drop the compute copy; backward re-fetches
            xin[r][u.index], outs[r][u.index], leaves[r][u.index] = x_in, out, leaf
            h[r] = out
        del full16, leaf                               # no local reference may outlive the release
        release_unit_params(ranks, u, "fwd")

    losses = [outs[r][U - 1].item() for r in range(N)]
    # what each rank holds of the parameters between forward and backward
    held_between = [sum(g.get("param16", u.index).nbytes for u in layout) + g.temp_bytes for g in ranks]
    assert all(leaves[r][ui].untyped_storage().size() == 0 for r in range(N) for ui in range(U))

    # ---------------- backward, unit by unit (last first)
    grad_h = [None] * N
    for u in reversed(layout):
        full16 = fetch_unit_params(ranks, u, log, step, "bwd", faults)
        local16 = []
        for r, g in enumerate(ranks):
            refetched[(r, u.index)] = full16[r].float()
            fc = FlopCounterMode(display=False) if r == probe_rank else None
            if fc:
                with fc:
                    torch.autograd.backward(outs[r][u.index], grad_h[r])
                probe["flops_bwd"] += fc.get_total_flops()
            else:
                torch.autograd.backward(outs[r][u.index], grad_h[r])
            del refetched[(r, u.index)]
            grad_h[r] = xin[r][u.index].grad if u.kind != "embed" else None
            g32 = leaves[r][u.index].grad
            local16.append((g32 / N).to(torch.bfloat16))     # pre-divide: the sum below is the mean
            leaves[r][u.index] = outs[r][u.index] = xin[r][u.index] = None
        # the unit's gradient now exists while its gathered weights are still held; reduce, then free both
        reduce_unit_grads(ranks, u, local16, log, step, faults)
        del full16, local16, g32
        release_unit_params(ranks, u, "bwd")

    # ---------------- global grad norm (clipping needs the norm of the *averaged* gradient)
    if stage == 0:
        # every rank has the full averaged gradient, so each computes the norm itself (32 times)
        sq = [sum(float(g.get("grad16", u.index).float().pow(2).sum()) for u in layout) for g in ranks]
        norms = [s ** 0.5 for s in sq]
    else:
        part = []
        for g in ranks:
            s = 0.0
            for u in layout:
                s += float(owned_grad(g, u.index).float().pow(2).sum())
            part.append(s)
        norms = [float(t) ** 0.5 for t in all_reduce_scalar(part, log, tag="grad_norm", step=step)]
    coefs = [min(1.0, opt.clip / (n + 1e-6)) for n in norms]

    # ---------------- optimizer: full tensor on DP, owned shard only on ZeRO
    t = step + 1
    for g, coef in zip(ranks, coefs):
        for u in layout:
            adam_update(g.get("master32", u.index), g.get("adam_m", u.index), g.get("adam_v", u.index),
                        owned_grad(g, u.index), t, coef, opt)
            g.optimizer_elements += g.get("master32", u.index).numel()

    # ---------------- make the new bf16 weights available where the stage needs them
    for u in layout:
        new16 = [g.get("master32", u.index).to(torch.bfloat16) for g in ranks]
        if stage == 0 or stage == 3:
            for g, w in zip(ranks, new16):        # DP: full copy; Z3: own shard. No communication.
                g.get("param16", u.index).copy_(w)
        else:
            if "skip_param_allgather" in faults:
                for g, w in zip(ranks, new16):
                    lo, hi = g.owned_range(u.index)
                    g.get("param16", u.index)[lo:hi].copy_(w)
            else:
                full = ring_all_gather(new16, log, tag="param_allgather_after_update", unit=u.name, step=step)
                for g, w in zip(ranks, full):
                    g.get("param16", u.index).copy_(w)

    return dict(losses=losses, grad_norm=norms[0], clip_coef=coefs[0], held_params_between_fwd_bwd=held_between,
                activation_bytes=act_bytes, probe=probe)


def owned_grad(g, ui):
    """The averaged gradient elements this rank is responsible for updating."""
    buf = g.get("grad16", ui)
    if g.stage == 0:
        return buf
    if g.stage == 1:
        lo, hi = g.owned_range(ui)
        return buf[lo:hi]
    return buf                              # Z2/Z3: the buffer *is* the shard


def reduce_unit_grads(ranks, u, local16, log, step, faults):
    stage = ranks[0].stage
    if stage == 0:
        if "skip_grad_sync" in faults:
            reduced = local16
        else:
            reduced = ring_all_reduce(local16, log, tag="grad_allreduce", unit=u.name, step=step)
        for g, t in zip(ranks, reduced):
            g.get("grad16", u.index).copy_(t)
    elif stage == 1:
        for g, t in zip(ranks, local16):     # full local gradient lives in the persistent buffer
            g.get("grad16", u.index).copy_(t)
        shards = ring_reduce_scatter(local16, log, tag="grad_reduce_scatter", unit=u.name, step=step)
        for g, s in zip(ranks, shards):
            lo, hi = g.owned_range(u.index)
            g.get("grad16", u.index)[lo:hi].copy_(s)
    else:
        for g, t in zip(ranks, local16):     # the full gradient exists only as a temporary bucket
            g.hold_temp(f"grad_bucket:{u.name}", t, event=f"grad bucket {u.name}")
        shards = ring_reduce_scatter(local16, log, tag="grad_reduce_scatter", unit=u.name, step=step)
        for g, s in zip(ranks, shards):
            g.get("grad16", u.index).copy_(s)
            g.release_temp(f"grad_bucket:{u.name}", event=f"free grad bucket {u.name}")


def adam_update(master, m, v, g16, t, coef, opt: OptConfig):
    g = g16.float() * coef
    m.mul_(opt.beta1).add_(g, alpha=1 - opt.beta1)
    v.mul_(opt.beta2).addcmul_(g, g, value=1 - opt.beta2)
    mhat = m / (1 - opt.beta1 ** t)
    vhat = v / (1 - opt.beta2 ** t)
    master.addcdiv_(mhat, vhat.sqrt().add_(opt.eps), value=-opt.lr)


# ---------------------------------------------------------------- driver
def run_stage(stage, cfg: GPTConfig, opt: OptConfig, init_flats, batches, world=32,
              faults=frozenset(), probe_rank=11, keep_cluster=False):
    layout = build_layout(cfg)
    if probe_rank is not None and probe_rank >= world:
        probe_rank = None
    ranks = make_cluster(stage, world, layout)
    load_initial_state(ranks, init_flats)
    log = CommLog(world)
    persistent = [g.persistent_bytes(by_category=True) for g in ranks]
    steps = []
    t0 = time.perf_counter()
    for step, (x, y) in enumerate(batches):
        for g, b in zip(ranks, split_batch(x, y, world)):
            g.batch = b
        steps.append(train_step(ranks, cfg, opt, step, log, faults, probe_rank))
        if step == 0:
            first_grad = reconstruct_grad(ranks)
            master_after_step1 = reconstruct(ranks)["master32"]
    wall = time.perf_counter() - t0
    # persistent state must not change size during training
    assert [g.persistent_bytes(by_category=True) for g in ranks] == persistent

    res = dict(stage=stage, name=STAGE_NAMES[stage], world=world, steps=steps, log=log, wall_s=wall,
               persistent_by_rank=persistent,
               peak_temp_by_rank=[g.peak_temp_bytes for g in ranks],
               optimizer_elements_by_rank=[g.optimizer_elements // len(batches) for g in ranks],
               temp_trace_probe=list(ranks[probe_rank].temp_trace) if probe_rank is not None else None,
               final=reconstruct(ranks), master_after_step1=master_after_step1, first_grad=first_grad, replica_max_diff=replica_divergence(ranks))
    if keep_cluster:
        res["ranks"] = ranks
    return res


def reconstruct(ranks):
    """Inspection only (not a collective): rebuild full fp32 master weights and bf16 weights per unit."""
    out = {"master32": [], "param16_rank0": []}
    for u in ranks[0].layout:
        if ranks[0].is_sharded("master32"):
            out["master32"].append(torch.cat([g.get("master32", u.index) for g in ranks]))
        else:
            out["master32"].append(ranks[0].get("master32", u.index).clone())
        if ranks[0].is_sharded("param16"):
            out["param16_rank0"].append(torch.cat([g.get("param16", u.index) for g in ranks]))
        else:
            out["param16_rank0"].append(ranks[0].get("param16", u.index).clone())
    return out


def reconstruct_grad(ranks):
    """Inspection only: the averaged gradient each owner used, stitched together (rank 0's copy on DP)."""
    if ranks[0].stage == 0:
        return [ranks[0].get("grad16", u.index).float().clone() for u in ranks[0].layout]
    return [torch.cat([owned_grad(g, u.index).float() for g in ranks]) for u in ranks[0].layout]


def replica_divergence(ranks):
    """Largest difference between any rank's copy and rank 0's copy, for every replicated category."""
    out = {}
    for cat in ("param16", "master32", "adam_m", "adam_v"):
        if ranks[0].is_sharded(cat):
            continue
        worst = 0.0
        for g in ranks[1:]:
            for u in ranks[0].layout:
                d = (g.get(cat, u.index).float() - ranks[0].get(cat, u.index).float()).abs().max()
                worst = max(worst, float(d))
        out[cat] = worst
    return out


# ---------------------------------------------------------------- single-device reference
def run_single_device(cfg: GPTConfig, opt: OptConfig, init_flats, batches):
    """One 'big GPU' that sees the whole global batch. No ranks, no collectives, plain autograd."""
    layout = build_layout(cfg)
    master = [f.clone() for f in init_flats]
    m = [torch.zeros_like(f) for f in init_flats]
    v = [torch.zeros_like(f) for f in init_flats]
    first_grad, losses = None, []
    for step, (x, y) in enumerate(batches):
        leaves = [w.to(torch.bfloat16).float().requires_grad_() for w in master]
        h = x
        for u in layout:
            h = unit_forward(u, cfg, h, u.views(leaves[u.index]))
        loss = lm_loss(h, y)
        loss.backward()
        losses.append(loss.item())
        g16 = [l.grad.to(torch.bfloat16) for l in leaves]
        if first_grad is None:
            first_grad = [g.float() for g in g16]
            first_grad_fp32 = [l.grad.clone() for l in leaves]
        norm = sum(float(g.float().pow(2).sum()) for g in g16) ** 0.5
        coef = min(1.0, opt.clip / (norm + 1e-6))
        for i in range(len(layout)):
            adam_update(master[i], m[i], v[i], g16[i], step + 1, coef, opt)
    return dict(master32=master, losses=losses, first_grad=first_grad, first_grad_fp32=first_grad_fp32)


def local_gradients(cfg: GPTConfig, init_flats, x, y, world):
    """fp32 gradient of each rank's own micro-batch loss (whole model flattened). Used for the collectives demo."""
    layout = build_layout(cfg)
    out = []
    for xb, yb in split_batch(x, y, world):
        leaves = [w.to(torch.bfloat16).float().requires_grad_() for w in init_flats]
        h = xb
        for u in layout:
            h = unit_forward(u, cfg, h, u.views(leaves[u.index]))
        lm_loss(h, yb).backward()
        out.append(torch.cat([l.grad for l in leaves]))
    return out
