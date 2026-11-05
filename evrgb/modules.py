"""Two-stem gated fusion: RGB stem + event stem (YOLO layers 0-3 each), per-pixel sigmoid gate at stride 8.

    x [B, 13, H, W] = [RGB(3) | events(10)]
      rgb -> rgb_stem -> f_rgb [B, C, H/8, W/8]
      evt -> evt_stem -> evt_mem (identity, slot for a future ConvLSTM) -> f_evt
      g = sigmoid(conv3x3(conv1x1(cat(f_rgb, f_evt))))
      fused = g * f_rgb + (1 - g) * f_evt
    returns [fused, f_rgb, f_evt, g]; the YAML picks `fused` with `Index [C, 0]`.
"""

from __future__ import annotations

from copy import deepcopy

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
        fuse_p2: bool = False,
    ):
        super().__init__()
        self.c_rgb, self.c_evt = c_rgb, c_in - c_rgb
        self.p_drop_rgb, self.p_drop_evt = p_drop_rgb, p_drop_evt
        self.rgb_stem = _stem(c_rgb, c_out, 1.0, n)
        self.evt_stem = _stem(self.c_evt, c_out, evt_width, n)
        self.evt_mem = nn.Identity()  # slot for a temporal block (ConvLSTM) later
        self.fuse = GatedFuse(c_out)
        # Optional second gate on the stride-4 maps (for a P2 detection head). Appends fused_p2 to the output list.
        self.fuse_p2 = GatedFuse(c_out) if fuse_p2 else None
        if fuse_p2:
            assert evt_width == 1.0, "fuse_p2 needs equal stride-4 widths in both stems (evt_width=1.0)"

    @property
    def mem_channels(self) -> int:
        return self.evt_stem[3].conv.out_channels

    def _drop(self, x: torch.Tensor, p: float) -> torch.Tensor:
        return _drop(self, x, p)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        rgb = self._drop(x[:, : self.c_rgb], self.p_drop_rgb)
        evt = self._drop(x[:, self.c_rgb :], self.p_drop_evt)
        p2_rgb = self.rgb_stem[:3](rgb)  # stride 4
        p2_evt = self.evt_stem[:3](evt)
        f_rgb = self.rgb_stem[3](p2_rgb)  # stride 8
        f_evt = self.evt_mem(self.evt_stem[3](p2_evt))
        fused, g = self.fuse(f_rgb, f_evt)
        out = [fused, f_rgb, f_evt, g]
        if self.fuse_p2 is not None:
            out.append(self.fuse_p2(p2_rgb, p2_evt)[0])
        return out


def _drop(module: nn.Module, x: torch.Tensor, p: float) -> torch.Tensor:
    """Modality dropout: zero the whole sample with probability p (train mode only)."""
    if not module.training or p <= 0:
        return x
    keep = (torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= p).to(x.dtype)
    return x * keep


# Official yolo26 backbone (full-width units) and the layers whose outputs are P3, P4, P5.
YOLO26_BACKBONE = [
    [-1, 1, "Conv", [64, 3, 2]],
    [-1, 1, "Conv", [128, 3, 2]],
    [-1, 2, "C3k2", [256, False, 0.25]],
    [-1, 1, "Conv", [256, 3, 2]],
    [-1, 2, "C3k2", [512, False, 0.25]],
    [-1, 1, "Conv", [512, 3, 2]],
    [-1, 2, "C3k2", [512, True]],
    [-1, 1, "Conv", [1024, 3, 2]],
    [-1, 2, "C3k2", [1024, True]],
    [-1, 1, "SPPF", [1024, 5, 3, True]],
    [-1, 2, "C2PSA", [1024]],
]
YOLO26_SCALES = {"n": [0.50, 0.25, 1024], "s": [0.50, 0.50, 1024], "m": [0.50, 1.00, 512], "l": [1.00, 1.00, 512], "x": [1.00, 1.50, 512]}
TOWER_TAPS = (4, 6, 10)


def _tower(c_in: int, scale: str | list | tuple) -> nn.Sequential:
    """One yolo26 backbone. `scale`: scale name or its [depth, width, max_channels] triple (a yaml arg must be the
    triple: parse_model replaces string args that happen to be its local names, e.g. "n", "m")."""
    from ultralytics.nn.tasks import parse_model

    triple = YOLO26_SCALES[scale] if isinstance(scale, str) else list(scale)
    d = {"nc": 1, "scale": "t", "scales": {"t": triple}, "backbone": deepcopy(YOLO26_BACKBONE), "head": []}
    seq, _ = parse_model(d, ch=c_in, verbose=False)
    return seq


class TwoTowerFuse(nn.Module):
    """Two complete YOLO26 backbones (RGB, events) with a gate per pyramid level.

        rgb -> rgb_tower -> (P3, P4, P5)_rgb
        evt -> evt_tower -> (P3 -> evt_mem, P4, P5)_evt
        fused_k = g_k * rgb_k + (1 - g_k) * evt_k   for k in P3, P4, P5
    returns [fused_P3, fused_P4, fused_P5, g_P3, g_P4, g_P5]; the YAML picks the fused maps with `Index [c, k]`.
    `scale` is the yolo26 scale name or its [depth, width, max_channels] triple; evrgb.config.load_dualstem_cfg fills
    the triple of the yaml's scale into the args.
    """

    def __init__(self, c_in: int = 13, c_rgb: int = 3, scale="n", p_drop_rgb: float = 0.0, p_drop_evt: float = 0.0):
        super().__init__()
        self.c_rgb, self.c_evt = c_rgb, c_in - c_rgb
        self.p_drop_rgb, self.p_drop_evt = p_drop_rgb, p_drop_evt
        self.rgb_tower = _tower(c_rgb, scale)
        self.evt_tower = _tower(self.c_evt, scale)
        self.evt_mem = nn.Identity()  # slot for a temporal block on the event P3 map
        chs = [_out_channels(self.evt_tower[i]) for i in TOWER_TAPS]
        self.fuse = nn.ModuleList(GatedFuse(c) for c in chs)

    @property
    def mem_channels(self) -> int:
        return _out_channels(self.evt_tower[TOWER_TAPS[0]])

    @staticmethod
    def _run(tower: nn.Sequential, x: torch.Tensor) -> list[torch.Tensor]:
        taps = []
        for i, m in enumerate(tower):
            x = m(x)
            if i in TOWER_TAPS:
                taps.append(x)
        return taps

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        rgb = _drop(self, x[:, : self.c_rgb], self.p_drop_rgb)
        evt = _drop(self, x[:, self.c_rgb :], self.p_drop_evt)
        f_rgb = self._run(self.rgb_tower, rgb)
        f_evt = self._run(self.evt_tower, evt)
        f_evt[0] = self.evt_mem(f_evt[0])
        fused, gates = [], []
        for fuse, r, e in zip(self.fuse, f_rgb, f_evt):
            o, g = fuse(r, e)
            fused.append(o)
            gates.append(g)
        return fused + gates


def _out_channels(m: nn.Module) -> int:
    return m.cv2.conv.out_channels if hasattr(m, "cv2") else m.conv.out_channels


def register() -> None:
    """Make the modules resolvable by name in Ultralytics' parse_model (it looks up ultralytics.nn.tasks globals)."""
    import ultralytics.nn.tasks as tasks

    tasks.DualStemFuse = DualStemFuse
    tasks.TwoTowerFuse = TwoTowerFuse
    tasks.GatedFuse = GatedFuse
