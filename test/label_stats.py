"""Box-size distribution of a dataset split (pixels). usage: python test/label_stats.py <split_dir>"""

import sys
from pathlib import Path

import numpy as np


def main(split_dir: str):
    files = sorted(Path(split_dir).glob("*/labels/*.npy"))
    labels = [np.load(f) for f in files]
    w = np.concatenate([l["w"] for l in labels]).astype(float)
    h = np.concatenate([l["h"] for l in labels]).astype(float)
    side = np.sqrt(w * h)
    n_frames, n_boxes = len(files), len(w)
    print(f"frames: {n_frames:,}   boxes: {n_boxes:,}   boxes/frame: {n_boxes / n_frames:.2f}   "
          f"frames without box: {100 * sum(len(l) == 0 for l in labels) / n_frames:.1f} %")
    print(f"{'':10s}{'p5':>7}{'p25':>7}{'p50':>7}{'p75':>7}{'p95':>7}{'max':>7}")
    for name, v in [("width", w), ("height", h), ("sqrt(wh)", side)]:
        q = np.percentile(v, [5, 25, 50, 75, 95])
        print(f"{name:10s}" + "".join(f"{x:7.1f}" for x in q) + f"{v.max():7.1f}")
    print("boxes with sqrt(wh) below:  " + "  ".join(f"{t} px: {100 * (side < t).mean():.1f} %" for t in (8, 16, 32, 64)))


if __name__ == "__main__":
    main(sys.argv[1])
