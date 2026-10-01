"""CPU check of the dual-stem weight remap. usage (repo root): PYTHONPATH=. python test/test_dualstem_weights.py <yolo26n.pt>"""

import sys

import torch

import evrgb  # noqa: F401
from evrgb.config import load_dualstem_cfg
from evrgb.weights import is_single_stem, load_pretrained
from ultralytics.nn.tasks import DetectionModel, load_checkpoint


def main(ckpt_path: str):
    m = DetectionModel(load_dualstem_cfg("conf/yolo26n_evrgb_dualstem.yaml"), ch=13, nc=1, verbose=False)
    src, _ = load_checkpoint(ckpt_path)
    src_sd = src.float().state_dict()
    assert is_single_stem(src_sd) and not is_single_stem(m.state_dict())

    report = load_pretrained(m, ckpt_path)
    bad = [k for k in report["skipped"] if ".cv3." not in k and ".one2one_cv3." not in k]
    assert not bad, f"unexpected skips: {bad[:5]}"
    sd = m.state_dict()
    assert torch.equal(sd["model.0.rgb_stem.0.conv.weight"], src_sd["model.0.conv.weight"])
    assert torch.equal(sd["model.2.cv1.conv.weight"], src_sd["model.4.cv1.conv.weight"])
    assert torch.equal(sd["model.21.cv2.0.0.conv.weight"], src_sd["model.23.cv2.0.0.conv.weight"])
    w_evt = sd["model.0.evt_stem.0.conv.weight"]
    assert w_evt.shape[1] == 10 and torch.allclose(w_evt.sum(1), src_sd["model.0.conv.weight"].sum(1))
    print(f"ok: {report['loaded']}/{report['total']} loaded, {len(report['skipped'])} skipped (all class-branch), "
          f"event first conv {tuple(w_evt.shape)}")


if __name__ == "__main__":
    main(sys.argv[1])
