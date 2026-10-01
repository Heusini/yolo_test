"""Write a tiny fake dataset in the real on-disk layout for CPU tests.

usage: python test/make_fake_dataset.py <out_dir>   -> <out_dir>/{train,val}/seq*/{events,labels,rgbs}/*.npy
"""

import sys
from pathlib import Path

import numpy as np

H, W, EV_CH, N_SEQ, N_FRAMES = 360, 640, 10, 2, 6
LABEL_DTYPE = [("x", "f4"), ("y", "f4"), ("w", "f4"), ("h", "f4"), ("class_id", "i8")]


def write_split(root: Path, rng: np.random.Generator):
    for s in range(N_SEQ):
        seq = root / f"seq{s}"
        for sub in ("events", "labels", "rgbs"):
            (seq / sub).mkdir(parents=True, exist_ok=True)
        for i in range(N_FRAMES):
            active = rng.random((EV_CH, H, W)) < 0.01
            events = active * rng.integers(1, 11, (EV_CH, H, W), dtype=np.uint8)
            rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
            n_box = rng.integers(0, 3)
            labels = np.zeros(n_box, dtype=LABEL_DTYPE)
            for b in range(n_box):
                w, h = rng.integers(8, 40), rng.integers(8, 40)
                labels[b] = (rng.integers(0, W - w), rng.integers(0, H - h), w, h, 0)
            np.save(seq / "events" / f"event_{i}.npy", events.astype(np.uint8))
            np.save(seq / "rgbs" / f"rgb_{i}.npy", rgb)
            np.save(seq / "labels" / f"label_{i}.npy", labels)


def main(out_dir: str):
    out = Path(out_dir)
    rng = np.random.default_rng(0)
    write_split(out / "train", rng)
    write_split(out / "val", rng)
    (out / "fake_data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train\nval: val\nchannels: {3 + EV_CH}\nimgsz: [384, 640]\nnames:\n  0: drone\n"
    )
    print(f"fake dataset written to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
