"""center_hits: a prediction counts when its center is inside a true box of its class, each true box at most once,
claimed in confidence order; box size is irrelevant. Then a CPU val on the fake set must report metrics/cAP(B) >= mAP50.
usage (repo root): PYTHONPATH=. python test/test_center_ap.py <fake_data.yaml> <best.pt>"""

import sys

import numpy as np
import torch

from engine.eventrgbvalidator import EventRGBValidator

gt = torch.tensor([[100.0, 100.0, 110.0, 110.0], [300.0, 200.0, 340.0, 240.0]])
gt_cls = torch.tensor([0.0, 0.0])
pred = torch.tensor([
    [90.0, 90.0, 120.0, 120.0],    # huge box, center (105, 105) inside gt0 -> hit (IoU would be 0.11)
    [104.0, 104.0, 106.0, 106.0],  # center (105, 105) inside gt0 too, lower confidence -> gt0 taken -> miss
    [330.0, 230.0, 360.0, 260.0],  # center (345, 245) outside gt1 -> miss
    [318.0, 218.0, 322.0, 222.0],  # center (320, 220) = gt1 center -> hit
])
conf = torch.tensor([0.9, 0.3, 0.8, 0.7])
hit = EventRGBValidator.center_hits(torch.zeros(4), gt_cls, pred, gt, conf)
assert hit.tolist() == [True, False, False, True], hit
assert EventRGBValidator.center_hits(torch.ones(4), gt_cls, pred, gt, conf).sum() == 0  # wrong class
# the confident prediction takes the true box even when a junk duplicate has the closer center
p2 = torch.tensor([[100.0, 100.0, 116.0, 116.0], [103.0, 103.0, 107.0, 107.0]])  # centers (108, 108) and (105, 105)
assert EventRGBValidator.center_hits(torch.zeros(2), gt_cls, p2, gt, torch.tensor([0.9, 0.01])).tolist() == [True, False]
assert EventRGBValidator.center_hits(torch.zeros(2), gt_cls, p2, gt, torch.tensor([0.01, 0.9])).tolist() == [False, True]
# one prediction inside two true boxes takes the closer one, the next prediction gets the other
gt2 = torch.tensor([[100.0, 100.0, 140.0, 140.0], [120.0, 120.0, 160.0, 160.0]])
p3 = torch.tensor([[126.0, 126.0, 130.0, 130.0], [124.0, 124.0, 132.0, 132.0]])  # centers (128, 128) and (128, 128)
assert EventRGBValidator.center_hits(torch.zeros(2), gt_cls, p3, gt2, torch.tensor([0.9, 0.8])).tolist() == [True, True]
assert EventRGBValidator.center_hits(torch.zeros(0), gt_cls, torch.zeros(0, 4), gt, torch.zeros(0)).shape == (0,)
print("ok: center_hits")

if len(sys.argv) > 2:
    from ultralytics import YOLO

    import importlib.util

    spec = importlib.util.spec_from_file_location("robustness_eval", "test/robustness_eval.py")  # "test" shadows the stdlib
    robustness_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(robustness_eval)
    RobustValidator = robustness_eval.RobustValidator  # clean condition, standalone dataloader

    m = YOLO(sys.argv[2]).val(data=sys.argv[1], imgsz=640, device="cpu", batch=4, workers=0, rect=True, plots=False,
                              validator=RobustValidator, project="runs/test", name="center_ap", exist_ok=True)
    print(f"mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}  cAP {m.center_ap:.3f}")
    assert not np.isnan(m.center_ap) and m.center_ap >= m.box.map50 - 1e-6
    print("ok: val reports cAP")
