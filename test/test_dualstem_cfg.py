"""CPU check of the dual-stem yaml for every scale. usage (repo root): PYTHONPATH=. python test/test_dualstem_cfg.py"""

import torch

import evrgb  # noqa: F401  (registers modules)
from evrgb.config import load_dualstem_cfg
from ultralytics.nn.modules import Index
from ultralytics.nn.tasks import DetectionModel

EXPECT_C = {"n": 64, "s": 128, "m": 256, "l": 256, "x": 384}


def main():
    x = torch.zeros(1, 13, 384, 640)
    for scale, c in EXPECT_C.items():
        cfg = load_dualstem_cfg(f"conf/yolo26{scale}_evrgb_dualstem.yaml")
        stem_args, index_args = cfg["backbone"][0][3], cfg["backbone"][1][3]
        assert stem_args[2] == index_args[0] == c, (scale, stem_args, index_args)
        m = DetectionModel(cfg, ch=13, nc=1, verbose=False).eval()
        assert isinstance(m.model[1], Index) and m.end2end and m.model[-1].reg_max == 1
        assert m.stride.tolist() == [8.0, 16.0, 32.0], m.stride
        with torch.no_grad():
            out = m(x)
        out = out[0] if isinstance(out, (tuple, list)) else out
        n_params = sum(p.numel() for p in m.parameters())
        print(f"{scale}: c_out={c:3d} stem repeats={stem_args[4]} out={tuple(out.shape)} params={n_params/1e6:.2f}M")


if __name__ == "__main__":
    main()
