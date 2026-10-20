"""ConvLSTM memory for the event stem (`DualStemFuse.evt_mem`), RVT style.

TemporalBlock sees a batch of B*L images = B sequence slots x L consecutive frames (slot-major, see
datasets/clip_sampler.py), loops over the L steps and keeps (h, c) per slot across calls, detached (TBPTT over L).
State lives in plain attributes (not buffers): invisible to state_dict / EMA / checkpoints; recreated as zeros
whenever its shape or device does not match the input (batch-size change, dummy forwards, .half()).
Output = x + out(h) with `out` zero-initialised, so a freshly inserted block is the identity.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, c: int, k: int = 3):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, padding=k // 2, groups=c)  # spatial mixing of the hidden state only
        self.gates = nn.Conv2d(2 * c, 4 * c, 1)
        with torch.no_grad():
            self.gates.bias[c : 2 * c].fill_(1.0)  # forget gate open at init: memory persists

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        i, f, g, o = self.gates(torch.cat([x, self.dw(h)], 1)).chunk(4, 1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class TemporalBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.cell = ConvLSTMCell(c)
        self.out = nn.Conv2d(c, c, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.clip_len = 1
        self._h = self._c = None

    def clear(self) -> None:
        self._h = self._c = None

    def reset(self, new_seq: torch.Tensor | None = None) -> None:
        """Zero the state of the flagged slots (bool [B]); None = all. No-op before the first forward."""
        if self._h is None:
            return
        if new_seq is None:
            self.clear()
            return
        keep = (~new_seq.to(self._h.device)).to(self._h.dtype).view(-1, 1, 1, 1)
        self._h, self._c = self._h * keep, self._c * keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = self.clip_len
        n, C, H, W = x.shape
        assert n % L == 0, f"batch {n} not a multiple of clip_len {L}"
        B = n // L
        if self._h is None or self._h.shape != (B, C, H, W) or self._h.device != x.device:
            self._h = x.new_zeros(B, C, H, W)
            self._c = x.new_zeros(B, C, H, W)
        h, c = self._h.to(x.dtype), self._c.to(x.dtype)
        xs = x.view(B, L, C, H, W)
        hs = []
        for t in range(L):
            h, c = self.cell(xs[:, t], h, c)
            hs.append(h)
        self._h, self._c = h.detach(), c.detach()
        return x + self.out(torch.stack(hs, 1).view(n, C, H, W))
