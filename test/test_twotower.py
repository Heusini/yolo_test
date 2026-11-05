"""CPU checks of the two-tower model: build, FLOPs vs dual stem, weight remaps, 1-epoch run, fresh reload, ONNX.

usage (repo root): PYTHONPATH=. python test/test_twotower.py <fake_data.yaml> <yolo26n.pt> <runs dir>
"""

import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

import evrgb  # noqa: F401
from engine.dualstem_trainer import DualStemTrainer
from evrgb import TwoTowerFuse
from evrgb.config import load_dualstem_cfg
from evrgb.weights import load_pretrained, stem_kind
from ultralytics.nn.modules import Index
from ultralytics.nn.tasks import DetectionModel, load_checkpoint


def build(name: str) -> DetectionModel:
    return DetectionModel(load_dualstem_cfg(f"conf/{name}.yaml"), ch=13, nc=1, verbose=False).eval()


def gflops(m: torch.nn.Module, x: torch.Tensor) -> float:
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(x)
    return fc.get_total_flops() / 1e9


def check_build():
    x = torch.zeros(1, 13, 384, 640)
    for scale, c3 in (("n", 128), ("s", 256)):
        m = build(f"yolo26{scale}_evrgb_twotower")
        assert isinstance(m.model[0], TwoTowerFuse) and all(isinstance(m.model[i], Index) for i in (1, 2, 3))
        assert m.stride.tolist() == [8.0, 16.0, 32.0] and m.end2end and m.model[-1].reg_max == 1
        with torch.no_grad():
            out = m(x)
        out = out[0] if isinstance(out, (tuple, list)) else out
        assert tuple(out.shape) == (1, 300, 6), out.shape
        maps = m.model[0](x)
        assert [t.shape[1] for t in maps[:3]] == [c3, c3, 2 * c3] and all(abs(float(g.mean()) - 0.5) < 1e-6 for g in maps[3:])
        print(f"{scale}: params {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M, GFLOPs@640x384 {gflops(m, x):.2f}")
    d = build("yolo26n_evrgb_dualstem")
    print(f"dual stem n: params {sum(p.numel() for p in d.parameters()) / 1e6:.2f}M, GFLOPs@640x384 {gflops(d, x):.2f}")
    # modality dropout with p=1 equals zeroing the RGB channels (train mode); eval mode never drops
    m = build("yolo26n_evrgb_twotower").train()
    m.model[0].p_drop_rgb = 1.0
    xr = torch.rand(2, 13, 64, 64)
    xz = xr.clone()
    xz[:, :3] = 0
    a = m.model[0](xr)[0]
    m.model[0].p_drop_rgb = 0.0
    b = m.model[0](xz)[0]
    assert torch.allclose(a, b), "dropout must zero the RGB input"
    m.model[0].p_drop_rgb = 1.0
    m.eval()
    with torch.no_grad():
        assert not torch.allclose(m.model[0](xr)[0], m.model[0](xz)[0]), "eval mode must not drop"


def check_remap_single(weights: str):
    m = build("yolo26n_evrgb_twotower")
    src_sd = load_checkpoint(weights)[0].float().state_dict()
    report = load_pretrained(m, weights)
    bad = [k for k in report["skipped"] if ".cv3." not in k and ".one2one_cv3." not in k]
    assert not bad, bad[:5]
    sd = m.state_dict()
    assert torch.equal(sd["model.0.rgb_tower.0.conv.weight"], src_sd["model.0.conv.weight"])
    assert torch.equal(sd["model.0.rgb_tower.10.cv1.conv.weight"], src_sd["model.10.cv1.conv.weight"])
    assert torch.equal(sd["model.0.evt_tower.10.cv1.conv.weight"], src_sd["model.10.cv1.conv.weight"])
    assert torch.equal(sd["model.6.cv1.conv.weight"], src_sd["model.13.cv1.conv.weight"])
    assert torch.equal(sd["model.16.cv2.0.0.conv.weight"], src_sd["model.23.cv2.0.0.conv.weight"])
    w = sd["model.0.evt_tower.0.conv.weight"]
    assert w.shape[1] == 10 and torch.allclose(w.sum(1), src_sd["model.0.conv.weight"].sum(1))
    print(f"single -> two tower: {report['loaded']}/{report['total']} loaded, {len(report['skipped'])} skipped (class branch)")


def check_remap_dual():
    src = build("yolo26n_evrgb_dualstem")
    for p in src.parameters():
        torch.nn.init.normal_(p)
    src_sd = src.state_dict()
    m = build("yolo26n_evrgb_twotower")
    report = load_pretrained(m, src_sd)
    assert stem_kind(src_sd) == "dualstem" and stem_kind(m.state_dict()) == "twotower"
    bad = [k for k in report["skipped"] if not k.startswith("model.0.fuse")]
    assert not bad, bad[:5]
    sd = m.state_dict()
    assert torch.equal(sd["model.0.rgb_tower.2.cv1.conv.weight"], src_sd["model.0.rgb_stem.2.cv1.conv.weight"])
    assert torch.equal(sd["model.0.evt_tower.0.conv.weight"], src_sd["model.0.evt_stem.0.conv.weight"])
    assert torch.equal(sd["model.0.rgb_tower.4.cv1.conv.weight"], src_sd["model.2.cv1.conv.weight"])
    assert torch.equal(sd["model.0.evt_tower.10.cv1.conv.weight"], src_sd["model.8.cv1.conv.weight"])
    assert torch.equal(sd["model.9.cv1.conv.weight"], src_sd["model.14.cv1.conv.weight"])
    assert torch.equal(sd["model.16.cv3.0.2.weight"], src_sd["model.21.cv3.0.2.weight"])
    assert float(sd["model.0.fuse.0.gate.weight"].abs().sum()) == 0, "gates must stay at their zero init"
    print(f"dual stem -> two tower: {report['loaded']}/{report['total']} loaded, {report["total"] - report["loaded"]} fresh (the three gates)")


def check_train(data: str, weights: str, runs: str):
    DualStemTrainer.MEM, DualStemTrainer.CLIP_LEN = "none", 1
    t = DualStemTrainer(overrides=dict(model="conf/yolo26n_evrgb_twotower.yaml", pretrained=weights, data=data, epochs=1,
                                       batch=4, device="cpu", workers=0, imgsz=640, rect=True, plots=False,
                                       project=runs, name="twotower", exist_ok=True))
    t.train()
    best = Path(t.best)
    code = (
        "import sys, torch; sys.path.insert(0, '.'); import evrgb\n"
        f"from ultralytics import YOLO; y = YOLO('{best}'); m = y.model.float().eval()\n"
        "assert type(m.model[0]).__name__ == 'TwoTowerFuse'\n"
        "out = m(torch.zeros(2, 13, 384, 640)); print('reloaded ok, output', tuple(out[0].shape))\n"
        f"p = y.export(format='onnx', imgsz=[384, 640], rect=True, nms=False)\n"
        "import onnxruntime as ort, numpy as np\n"
        "s = ort.InferenceSession(p, providers=['CPUExecutionProvider'])\n"
        "o = s.run(None, {s.get_inputs()[0].name: np.zeros((1, 13, 384, 640), np.float32)})[0]\n"
        "print('onnx ok, output', o.shape)\n"
    )
    print(subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout.strip())


def main(data, weights, runs):
    check_build()
    check_remap_single(weights)
    check_remap_dual()
    check_train(data, weights, runs)
    print("ok")


if __name__ == "__main__":
    main(*sys.argv[1:4])
