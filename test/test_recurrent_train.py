"""1-epoch CPU run of DualStemTrainer with the ConvLSTM + clip batches on the fake set, then reload best.pt fresh.

usage: PYTHONPATH=. python test/test_recurrent_train.py <fake_data.yaml> <yolo26n.pt> <runs dir>
"""

import subprocess
import sys
from pathlib import Path

import torch

import evrgb  # noqa: F401
from engine.dualstem_trainer import DualStemTrainer
from evrgb import temporal
from evrgb.temporal import TemporalBlock, temporal_blocks

RESETS = []
_reset = TemporalBlock.reset


def spy(self, new_seq=None):
    RESETS.append(None if new_seq is None else new_seq.tolist())
    return _reset(self, new_seq)


def main(data, weights, runs):
    TemporalBlock.reset = spy
    DualStemTrainer.MEM, DualStemTrainer.CLIP_LEN, DualStemTrainer.RANDOM_SLOTS = "lstm", 3, 0
    t = DualStemTrainer(overrides=dict(model="conf/yolo26n_evrgb_dualstem.yaml", pretrained=weights, data=data, epochs=1,
                                       batch=3, device="cpu", workers=0, imgsz=640, rect=True, plots=False,
                                       project=runs, name="recurrent", exist_ok=True, val=True))
    t.train()
    blocks = temporal_blocks(t.model)
    assert len(blocks) == 1 and blocks[0].clip_len == 1  # cleared before the last checkpoint
    assert blocks[0]._h is None, "state must be cleared before the checkpoint"
    flags = [r for r in RESETS if r is not None]
    assert flags and all(len(f) in (1, 2) for f in flags), flags  # train: 1 slot, val: 2 slots (6 frames / 3)
    assert any(all(f) for f in flags) and any(not all(f) for f in flags), "expected resets at sequence starts only"
    print("reset flags:", flags)

    best = Path(t.best)
    code = (
        "import sys, torch; sys.path.insert(0, '.'); import evrgb; from evrgb.temporal import temporal_blocks\n"
        f"from ultralytics import YOLO; m = YOLO('{best}').model\n"
        "b = temporal_blocks(m); assert len(b) == 1 and b[0]._h is None, b\n"
        "x = torch.zeros(2, 13, 384, 640); m.float().eval(); y = m(x)\n"
        "print('reloaded ok, output', tuple(y[0].shape))\n"
    )
    print(subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout.strip())

    # the default path must be untouched: stateless model, shuffled single-frame loader
    RESETS.clear()
    DualStemTrainer.MEM, DualStemTrainer.CLIP_LEN = "none", 1
    t = DualStemTrainer(overrides=dict(model="conf/yolo26n_evrgb_dualstem.yaml", pretrained=weights, data=data, epochs=1,
                                       batch=4, device="cpu", workers=0, imgsz=640, rect=True, plots=False,
                                       project=runs, name="stateless", exist_ok=True))
    t.train()
    assert not temporal_blocks(t.model) and not RESETS and isinstance(t.model.model[0].evt_mem, torch.nn.Identity)
    print("ok")


if __name__ == "__main__":
    main(*sys.argv[1:4])
