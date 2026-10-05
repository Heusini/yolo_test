"""Scale the DualStemFuse / Index literals of a dual-stem yaml the way parse_model scales its own modules."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils.ops import make_divisible


def load_dualstem_cfg(cfg: str | Path | dict) -> dict:
    """yaml path or loaded dict -> dict with DualStemFuse (c_out, repeats) and Index channels scaled for `scale`.

    Scaled Index lines: the one right after DualStemFuse and any Index whose `from` is the DualStemFuse layer.
    """
    d = deepcopy(cfg) if isinstance(cfg, dict) else deepcopy(yaml_model_load(cfg))
    if d.get("dualstem_resolved"):
        return d
    scale = d.get("scale") or next(iter(d["scales"]))
    depth, width, max_channels = d["scales"][scale]
    d["scale"] = scale

    def ch(c: int) -> int:
        return make_divisible(min(c, max_channels) * width, 8)

    layers = d["backbone"] + d["head"]
    dual = next(i for i, l in enumerate(layers) if l[2] == "DualStemFuse")
    args = layers[dual][3]
    args[2] = ch(args[2])
    args[4] = max(round(args[4] * depth), 1)
    for i, (f, _, module, a) in enumerate(layers):
        if module == "Index" and (i == dual + 1 or f == dual):
            a[0] = args[2]
    d["dualstem_resolved"] = True
    return d
