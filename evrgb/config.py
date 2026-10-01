"""Scale the DualStemFuse / Index literals of a dual-stem yaml the way parse_model scales its own modules."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils.ops import make_divisible


def load_dualstem_cfg(path: str | Path) -> dict:
    """yaml_model_load() + width/depth scaling of DualStemFuse (c_out, repeats) and the Index channel count."""
    d = deepcopy(yaml_model_load(path))
    scale = d.get("scale") or next(iter(d["scales"]))
    depth, width, max_channels = d["scales"][scale]
    d["scale"] = scale

    def ch(c: int) -> int:
        return make_divisible(min(c, max_channels) * width, 8)

    layers = d["backbone"]
    for i, (_, _, module, args) in enumerate(layers):
        if module == "DualStemFuse":
            args[2] = ch(args[2])
            args[4] = max(round(args[4] * depth), 1)
            if layers[i + 1][2] == "Index":
                layers[i + 1][3][0] = args[2]
    return d
