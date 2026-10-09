"""Figures for the notebook. Every function takes already-measured numbers; nothing is computed here."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .cluster import CATEGORIES, SHARDED, STAGE_NAMES

CAT_COLORS = {"param16": "#4C72B0", "grad16": "#DD8452", "master32": "#55A868", "adam_m": "#8172B3", "adam_v": "#937860"}
CAT_LABELS = {"param16": "bf16 weights (2 B)", "grad16": "bf16 grads (2 B)", "master32": "fp32 master (4 B)",
              "adam_m": "Adam m (4 B)", "adam_v": "Adam v (4 B)"}
STAGE_COLORS = {0: "#C44E52", 1: "#DD8452", 2: "#55A868", 3: "#4C72B0"}
MIB = 2 ** 20


def ownership_grid(world, highlight_rank, bytes_per_param, path):
    """Rows: stages. Columns: state categories. Each cell: rank (y) x shard id (x), filled if held."""
    fig, axes = plt.subplots(4, 5, figsize=(11, 8.6), sharex=True, sharey=True)
    for st in range(4):
        for j, cat in enumerate(CATEGORIES):
            ax = axes[st, j]
            img = np.eye(world) if cat in SHARDED[st] else np.ones((world, world))
            rgba = np.zeros((world, world, 4))
            rgba[img > 0] = plt.matplotlib.colors.to_rgba(CAT_COLORS[cat])
            rgba[img == 0] = (0.93, 0.93, 0.93, 1)
            ax.imshow(rgba, aspect="equal", interpolation="nearest")
            ax.axhline(highlight_rank, color="black", lw=0.8, alpha=0.7)
            ax.set_xticks([0, world - 1]); ax.set_yticks([0, highlight_rank, world - 1])
            if st == 0:
                ax.set_title(CAT_LABELS[cat], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{STAGE_NAMES[st]}\n{bytes_per_param[st]:.4g} B/param\nrank", fontsize=9)
            if st == 3:
                ax.set_xlabel("shard id", fontsize=8)
    fig.suptitle(f"Who holds what: {world} ranks x {world} shards per state tensor "
                 f"(coloured = kept between steps; line = rank {highlight_rank})", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return fig


def memory_per_rank(per_rank_by_cat, world, total_unique_bytes, path):
    """Left: one rank's persistent bytes by category. Right: summed over all ranks vs one copy of the state."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    stages = list(range(4))
    x = np.arange(4)
    bottom = np.zeros(4)
    for cat in CATEGORIES:
        vals = np.array([per_rank_by_cat[st][cat] / MIB for st in stages])
        a.bar(x, vals, bottom=bottom, color=CAT_COLORS[cat], label=CAT_LABELS[cat], width=0.6)
        bottom += vals
    for i, v in enumerate(bottom):
        a.text(i, v + 0.15, f"{v:.2f} MiB", ha="center", fontsize=9)
    a.set_xticks(x, [STAGE_NAMES[s] for s in stages]); a.set_ylabel("persistent state per rank (MiB)")
    a.set_title("One rank (measured tensor.nbytes)"); a.legend(fontsize=8)
    agg = [sum(per_rank_by_cat[st].values()) * world / MIB for st in stages]
    b.bar(x, agg, color=[STAGE_COLORS[s] for s in stages], width=0.6)
    b.axhline(total_unique_bytes / MIB, color="black", ls="--", lw=1)
    b.text(-0.45, total_unique_bytes / MIB * 0.80, "one copy of the training state", ha="left", fontsize=8)
    for i, v in enumerate(agg):
        b.text(i, v * 1.07, f"{v:.1f} MiB = {v / (total_unique_bytes / MIB):.3g} cop" + ("y" if abs(v - total_unique_bytes / MIB) < 1e-9 else "ies"), ha="center", fontsize=8)
    b.set_xticks(x, [STAGE_NAMES[s] for s in stages]); b.set_ylabel(f"summed over {world} ranks (MiB, log scale)")
    b.set_yscale("log"); b.set_ylim(8, 900); b.set_title("Whole cluster: how many copies exist")
    b.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    fig.tight_layout(); fig.savefig(path, dpi=130)
    return fig


def world_size_scaling(worlds, bpp, card_bpp_30b, gib_per_bpp, path):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for st in range(4):
        ax.plot(worlds, [bpp[st][n] for n in worlds], "o-", color=STAGE_COLORS[st], label=STAGE_NAMES[st])
    ax.axhline(4, color=STAGE_COLORS[1], ls=":", lw=1); ax.text(worlds[0], 4.25, "ZeRO-1 floor: 4 B (weights + grads)", fontsize=8)
    ax.axhline(2, color=STAGE_COLORS[2], ls=":", lw=1); ax.text(worlds[0], 1.83, "ZeRO-2 floor: 2 B (weights)", fontsize=8, va="top")
    ax.axhline(card_bpp_30b, color="black", ls="--", lw=1)
    ax.text(worlds[0], card_bpp_30b * 1.05, "74.5 GiB card, if the model is 30B (right axis)", fontsize=8)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(worlds, [str(n) for n in worlds])
    ax.set_xlabel("world size N (ranks)"); ax.set_ylabel("persistent bytes per parameter per rank")
    sec = ax.secondary_yaxis("right", functions=(lambda y: y * gib_per_bpp, lambda g: g / gib_per_bpp))
    sec.set_ylabel("same, for a 30B model (GiB per GPU)")
    ax.set_title("Per-rank training state vs world size (measured from rank construction)")
    ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=130)
    return fig


def communication(comm_rows, course_P, path):
    """comm_rows[stage] = {payload label: bytes per rank in units of P}."""
    fig, ax = plt.subplots(figsize=(8, 4.4))
    labels = sorted({k for r in comm_rows.values() for k in r})
    palette = dict(zip(labels, ["#C44E52", "#DD8452", "#55A868", "#4C72B0", "#8172B3", "#937860"]))
    x = np.arange(4); bottom = np.zeros(4)
    for lab in labels:
        vals = np.array([comm_rows[st].get(lab, 0.0) for st in range(4)])
        ax.bar(x, vals, bottom=bottom, color=palette[lab], label=lab, width=0.6)
        bottom += vals
    ax.scatter(x, [course_P[st] for st in range(4)], marker="_", s=900, color="black", zorder=5, label="course value (2P / 3P)")
    for i, v in enumerate(bottom):
        ax.text(i, max(v, course_P[i]) + 0.08, f"measured {v:.4g} P", ha="center", fontsize=9)
    ax.set_xticks(x, [STAGE_NAMES[s] for s in range(4)])
    ax.set_ylabel("bytes sent per rank per step (units of P = 2 B x params)")
    ax.set_title("Communication per step, measured from the ring send log")
    ax.legend(fontsize=8, loc="upper left"); ax.set_ylim(0, 3.6)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    return fig


def temp_timeline(trace, rank, persistent_param_bytes, path):
    """trace = list of (event, live temp bytes) for one ZeRO-3 rank over one step."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    y = [0] + [b / 1024 for _, b in trace]
    ax.step(range(len(y)), y, where="post", color=STAGE_COLORS[3], label="ZeRO-3: temporary buffers (gathered weights, gradient buckets)")
    for i, (ev, b) in enumerate(trace):
        if ev.startswith("+gather"):
            ax.annotate(ev[1:].replace("gather ", ""), (i + 1, b / 1024), fontsize=6.5, rotation=60,
                        xytext=(0, 4), textcoords="offset points")
    for st, pb in persistent_param_bytes.items():
        ax.axhline(pb / 1024, ls="--", lw=1, color=STAGE_COLORS[st],
                   label=f"{STAGE_NAMES[st]}: bf16 weights this rank keeps all the time")
    ax.set_xlabel("event number within one training step (forward: 6 gathers; backward: 6 gathers + 6 gradient buckets)")
    ax.set_ylabel("KiB held by the rank")
    ax.set_title(f"Rank {rank} under ZeRO-3: gather -> compute -> release, unit by unit")
    ax.legend(fontsize=7.5, loc="upper left"); ax.set_ylim(0, None)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    return fig
