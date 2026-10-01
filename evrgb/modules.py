"""Two-stem gated fusion: RGB stem + event stem (YOLO layers 0-3 each), per-pixel sigmoid gate at stride 8.

    x [B, 13, H, W] = [RGB(3) | events(10)]
      rgb -> rgb_stem -> f_rgb [B, C, H/8, W/8]
      evt -> evt_stem -> evt_mem (identity, slot for a future ConvLSTM) -> f_evt
      g = sigmoid(conv3x3(conv1x1(cat(f_rgb, f_evt))))
      fused = g * f_rgb + (1 - g) * f_evt
    returns [fused, f_rgb, f_evt, g]; the YAML picks `fused` with `Index [C, 0]`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules import C3k2, Conv
from ultralytics.utils.ops import make_divisible


class GatedFuse(nn.Module):
    def __init__(self, c: int, r: int = 4):
        super().__init__()
        ch = max(make_divisible(c // r, 8), 8)
        self.reduce = Conv(2 * c, ch, 1)
        self.gate = nn.Conv2d(ch, c, 3, padding=1)
        nn.init.zeros_(self.gate.weight)  # g = 0.5 at init -> plain average
        nn.init.zeros_(self.gate.bias)

    def forward(self, rgb: torch.Tensor, evt: torch.Tensor):
        g = torch.sigmoid(self.gate(self.reduce(torch.cat([rgb, evt], 1))))
        return g * rgb + (1.0 - g) * evt, g


def _stem(c_in: int, c_out: int, width: float, n: int) -> nn.Sequential:
    """YOLO26 layers 0-3: Conv s2, Conv s2, C3k2, Conv s2 -> stride 8."""
    c1 = make_divisible(c_out / 4 * width, 8)
    c2 = make_divisible(c_out / 2 * width, 8)
    c3 = make_divisible(c_out * width, 8)
    return nn.Sequential(
        Conv(c_in, c1, 3, 2),
        Conv(c1, c2, 3, 2),
        C3k2(c2, c3, n, False, 0.25),
        Conv(c3, c_out, 3, 2),
    )


class DualStemFuse(nn.Module):
    def __init__(
        self,
        c_in: int = 13,
        c_rgb: int = 3,
        c_out: int = 64,
        evt_width: float = 1.0,
        n: int = 1,
        p_drop_rgb: float = 0.0,
        p_drop_evt: float = 0.0,
    ):
        super().__init__()
        self.c_rgb, self.c_evt = c_rgb, c_in - c_rgb
        self.p_drop_rgb, self.p_drop_evt = p_drop_rgb, p_drop_evt
        self.rgb_stem = _stem(c_rgb, c_out, 1.0, n)
        self.evt_stem = _stem(self.c_evt, c_out, evt_width, n)
        self.evt_mem = nn.Identity()  # slot for a temporal block (ConvLSTM) later
        self.fuse = GatedFuse(c_out)

    def _drop(self, x: torch.Tensor, p: float) -> torch.Tensor:
        if not self.training or p <= 0:
            return x
        keep = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= p).to(x.dtype)
        return x * keep

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        rgb = self._drop(x[:, : self.c_rgb], self.p_drop_rgb)
        evt = self._drop(x[:, self.c_rgb :], self.p_drop_evt)
        f_rgb = self.rgb_stem(rgb)
        f_evt = self.evt_mem(self.evt_stem(evt))
        fused, g = self.fuse(f_rgb, f_evt)
        return [fused, f_rgb, f_evt, g]


def register() -> None:
    """Make the modules resolvable by name in Ultralytics' parse_model (it looks up ultralytics.nn.tasks globals)."""
    import ultralytics.nn.tasks as tasks

    tasks.DualStemFuse = DualStemFuse
    tasks.GatedFuse = GatedFuse
