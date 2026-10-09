"""A small GPT written as a sequence of "units" whose parameters live in flat buffers.

Each unit (embedding, block 0..3, final norm + head) owns one flat parameter
vector. Sharding, gathering and gradient reduction all operate on these flat
vectors, so a rank's slice of a unit is just a contiguous index range.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 256      # byte-level tokens
    block_size: int = 64       # context length
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ParamEntry:
    name: str
    shape: tuple
    offset: int

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class Unit:
    """One gather/release granule: a contiguous flat buffer of named tensors."""
    index: int
    name: str
    kind: str                 # "embed" | "block" | "head"
    entries: tuple            # tuple[ParamEntry, ...]

    @property
    def numel(self) -> int:
        return sum(e.numel for e in self.entries)

    def views(self, flat: torch.Tensor) -> dict:
        """Named tensor views into a flat buffer (no copies)."""
        return {e.name: flat[e.offset:e.offset + e.numel].view(e.shape) for e in self.entries}


def _unit(index, name, kind, specs):
    entries, off = [], 0
    for pname, shape in specs:
        entries.append(ParamEntry(pname, tuple(shape), off))
        off += math.prod(shape)
    return Unit(index, name, kind, tuple(entries))


def build_layout(cfg: GPTConfig) -> list[Unit]:
    d, v, t = cfg.d_model, cfg.vocab_size, cfg.block_size
    units = [_unit(0, "embed", "embed", [("wte", (v, d)), ("wpe", (t, d))])]
    for i in range(cfg.n_layer):
        units.append(_unit(len(units), f"block{i}", "block", [
            ("ln1.weight", (d,)), ("ln1.bias", (d,)),
            ("attn.qkv.weight", (3 * d, d)), ("attn.qkv.bias", (3 * d,)),
            ("attn.proj.weight", (d, d)), ("attn.proj.bias", (d,)),
            ("ln2.weight", (d,)), ("ln2.bias", (d,)),
            ("mlp.fc.weight", (4 * d, d)), ("mlp.fc.bias", (4 * d,)),
            ("mlp.proj.weight", (d, 4 * d)), ("mlp.proj.bias", (d,)),
        ]))
    units.append(_unit(len(units), "head", "head",
                       [("ln_f.weight", (d,)), ("ln_f.bias", (d,)), ("lm_head.weight", (v, d))]))
    return units


def init_flat_params(cfg: GPTConfig, layout: list[Unit], seed: int = 0) -> list[torch.Tensor]:
    """GPT-2 style init, returned as one fp32 flat vector per unit."""
    g = torch.Generator().manual_seed(seed)
    flats = []
    for unit in layout:
        flat = torch.empty(unit.numel, dtype=torch.float32)
        for e, view in zip(unit.entries, unit.views(flat).values()):
            leaf = e.name.split(".")[-1]
            if e.name.startswith(("ln", "ln_f")) and leaf == "weight":
                view.fill_(1.0)
            elif leaf == "bias":
                view.zero_()
            else:
                std = 0.02
                if e.name in ("attn.proj.weight", "mlp.proj.weight"):
                    std = 0.02 / math.sqrt(2 * cfg.n_layer)
                view.normal_(0.0, std, generator=g)
        flats.append(flat)
    return flats


def unit_forward(unit: Unit, cfg: GPTConfig, x, p: dict):
    """x is token ids for the embedding unit, hidden states otherwise. Returns hidden or logits."""
    if unit.kind == "embed":
        T = x.shape[1]
        return p["wte"][x] + p["wpe"][:T]
    if unit.kind == "block":
        B, T, C = x.shape
        h = F.layer_norm(x, (C,), p["ln1.weight"], p["ln1.bias"])
        q, k, v = F.linear(h, p["attn.qkv.weight"], p["attn.qkv.bias"]).split(C, dim=2)
        q, k, v = (z.view(B, T, cfg.n_head, C // cfg.n_head).transpose(1, 2) for z in (q, k, v))
        # explicit attention so every matmul is visible to the FLOP counter
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(C // cfg.n_head)
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        att = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1) @ v
        x = x + F.linear(att.transpose(1, 2).reshape(B, T, C), p["attn.proj.weight"], p["attn.proj.bias"])
        h = F.layer_norm(x, (C,), p["ln2.weight"], p["ln2.bias"])
        h = F.gelu(F.linear(h, p["mlp.fc.weight"], p["mlp.fc.bias"]))
        return x + F.linear(h, p["mlp.proj.weight"], p["mlp.proj.bias"])
    # head
    h = F.layer_norm(x, (x.shape[-1],), p["ln_f.weight"], p["ln_f.bias"])
    return F.linear(h, p["lm_head.weight"])


def lm_loss(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
