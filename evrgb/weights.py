"""Copy single-stem YOLO26 weights into the dual-stem model.

    yolo26 layers 0-3  -> model.0.rgb_stem.{0-3}  (verbatim)
                       -> model.0.evt_stem.{0-3}  (first conv: RGB kernels averaged, tiled over the event channels)
    yolo26 layers 4..  -> model.{i-2}             (shape mismatches, i.e. the nc-dependent class branch, are skipped)
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import LOGGER

_LAYER = re.compile(r"^model\.(\d+)\.(.*)$")
_DETECT_LEVEL = re.compile(r"^(cv2|cv3|one2one_cv2|one2one_cv3)\.(\d+)\.")
# yolo26 layer -> P2-variant layer: trunk and top-down neck shift by -2, bottom-up neck by +5 (P2 branch inserted),
# Detect 23 -> 28. The P2 branch itself (layers 15-21) has no pretrained counterpart.
_P2_MAP = {**{i: i - 2 for i in range(4, 17)}, 17: 22, 18: 23, 19: 24, 20: 25, 21: 26, 22: 27, 23: 28}


def is_single_stem(state_dict: dict) -> bool:
    return "model.0.conv.weight" in state_dict


def _adapt_first_conv(w: torch.Tensor, c_in_new: int) -> torch.Tensor:
    """[C_out, 3, k, k] -> [C_out, c_in_new, k, k] with the same response magnitude."""
    return w.mean(dim=1, keepdim=True).repeat(1, c_in_new, 1, 1) * (w.shape[1] / c_in_new)


def load_pretrained(model: nn.Module, src: str | Path | nn.Module | dict) -> dict:
    """Remap `src` (checkpoint path, module or state dict) into a DualStemFuse-based DetectionModel."""
    if isinstance(src, (str, Path)):
        src, _ = load_checkpoint(str(src))
    sd = src if isinstance(src, dict) else src.float().state_dict()
    dst = model.state_dict()

    p2 = len(model.model) == 29  # P2-head variant (conf/yolo26_evrgb_dualstem_p2.yaml) vs 22 layers for P3
    layer_map = _P2_MAP if p2 else {i: i - 2 for i in range(4, 24)}

    remapped = {}
    for k, v in sd.items():
        m = _LAYER.match(k)
        if not m:
            continue
        i, rest = int(m.group(1)), m.group(2)
        if i <= 3:
            remapped[f"model.0.rgb_stem.{i}.{rest}"] = v
            remapped[f"model.0.evt_stem.{i}.{rest}"] = v
        elif i in layer_map:
            if p2 and i == 23:  # Detect: yolo26 levels P3,P4,P5 -> ours P2,P3,P4,P5 (level index + 1)
                rest = _DETECT_LEVEL.sub(lambda mm: f"{mm.group(1)}.{int(mm.group(2)) + 1}.", rest)
            remapped[f"model.{layer_map[i]}.{rest}"] = v

    loaded, skipped = {}, []
    for k, v in remapped.items():
        if k not in dst:
            skipped.append(k)
            continue
        if v.shape != dst[k].shape:
            if k == "model.0.evt_stem.0.conv.weight" and v.shape[0] == dst[k].shape[0]:
                v = _adapt_first_conv(v, dst[k].shape[1])
            else:
                skipped.append(k)
                continue
        loaded[k] = v
    model.load_state_dict(loaded, strict=False)
    LOGGER.info(f"Transferred {len(loaded)}/{len(dst)} items from pretrained weights (dual-stem remap)")
    return {"loaded": len(loaded), "total": len(dst), "skipped": skipped}
