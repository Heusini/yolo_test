"""CPU check of the P2-head dual-stem variant. usage (repo root): PYTHONPATH=. python test/test_dualstem_p2.py <yolo26n.pt>"""

import sys

import torch

import evrgb  # noqa: F401
from evrgb import DualStemFuse
from evrgb.config import load_dualstem_cfg
from evrgb.weights import load_pretrained
from ultralytics.nn.modules import Index
from ultralytics.nn.tasks import DetectionModel


def main(ckpt_path: str):
    # module: 5th output = fused stride-4 map
    m = DualStemFuse(13, 3, 64, fuse_p2=True).eval()
    out = m(torch.randn(1, 13, 384, 640))
    assert len(out) == 5 and out[4].shape == (1, 64, 96, 160) and out[0].shape == (1, 64, 48, 80), [o.shape for o in out]
    assert len(DualStemFuse(13, 3, 64)(torch.randn(1, 13, 384, 640))) == 4, "P3-only module must be unchanged"

    for scale, c in [("n", 64), ("s", 128)]:
        cfg = load_dualstem_cfg(f"conf/yolo26{scale}_evrgb_dualstem_p2.yaml")
        layers = cfg["backbone"] + cfg["head"]
        idx = [l for l in layers if l[2] == "Index"]
        assert len(idx) == 2 and idx[0][3][0] == idx[1][3][0] == c, idx
        model = DetectionModel(cfg, ch=13, nc=1, verbose=False).eval()
        assert len(model.model) == 29 and isinstance(model.model[16], Index)
        assert model.stride.tolist() == [4.0, 8.0, 16.0, 32.0], model.stride
        with torch.no_grad():
            y = model(torch.zeros(1, 13, 384, 640))
        y = y[0] if isinstance(y, (tuple, list)) else y
        print(f"{scale}: strides {model.stride.tolist()} out {tuple(y.shape)} params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # weights: trunk + P3/P4/P5 neck + box branches of levels 1-3 come from yolo26n.pt; P2 branch stays random
    model = DetectionModel(load_dualstem_cfg("conf/yolo26n_evrgb_dualstem_p2.yaml"), ch=13, nc=1, verbose=False)
    r = load_pretrained(model, ckpt_path)
    skipped = r["skipped"]
    assert all(".cv3." in k or ".one2one_cv3." in k for k in skipped), [k for k in skipped if ".cv3." not in k][:5]
    sd = model.state_dict()
    src = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"].float().state_dict()
    assert torch.equal(sd["model.27.cv1.conv.weight"], src["model.22.cv1.conv.weight"]), "P5 block must map 22 -> 27"
    assert torch.equal(sd["model.28.cv2.1.0.conv.weight"], src["model.23.cv2.0.0.conv.weight"]), "Detect level shift"
    n_p2_new = sum(v.numel() for k, v in sd.items() if any(k.startswith(f"model.{i}.") for i in range(15, 22)))
    print(f"ok: {r['loaded']}/{r['total']} loaded, {len(skipped)} skipped (class branches only), "
          f"P2 branch random-init params {n_p2_new/1e3:.0f}k")


if __name__ == "__main__":
    main(sys.argv[1])
