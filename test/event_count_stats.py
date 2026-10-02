"""Distribution of event counts per bin over a dataset split, to choose the count cutoff.

usage (repo root): python test/event_count_stats.py <split_dir> [max_files]
  e.g. python test/event_count_stats.py /scratch/sheusinger/<dataset>/train 500
"""

import sys
from pathlib import Path

import numpy as np

PERCENTILES = [50, 90, 99, 99.9, 99.99]


def main(split_dir: str, max_files: int = 500):
    files = sorted(Path(split_dir).glob("*/events/*.npy"))
    if not files:
        sys.exit(f"no */events/*.npy under {split_dir}")
    step = max(1, len(files) // max_files)
    files = files[::step]

    hist = np.zeros(65536, dtype=np.int64)  # histogram of counts (0..65535)
    n_px, dtype, shape = 0, None, None
    for f in files:
        ev = np.load(f)
        dtype, shape = ev.dtype, ev.shape
        hist += np.bincount(ev.astype(np.int64).ravel(), minlength=65536)[:65536]
        n_px += ev.size

    nonzero = hist.copy()
    nonzero[0] = 0
    n_nz = nonzero.sum()
    cdf = np.cumsum(nonzero) / max(n_nz, 1)
    vmax = int(np.nonzero(hist)[0].max())

    print(f"files sampled: {len(files)} of {len(files) * step}, dtype {dtype}, shape {shape}")
    print(f"bins total: {n_px:,}   non-zero: {n_nz:,} ({100 * n_nz / n_px:.3f} %)")
    print(f"max count: {vmax}   bins at max: {hist[vmax]:,} ({100 * hist[vmax] / max(n_nz, 1):.3f} % of non-zero)")
    print("percentiles of non-zero counts:")
    for p in PERCENTILES:
        print(f"  p{p:<6} {int(np.searchsorted(cdf, p / 100) )}")
    print("fraction of non-zero bins that a cutoff would clip:")
    for c in [5, 10, 15, 20, 30, 50, 100, 255]:
        if c <= vmax:
            print(f"  cutoff {c:>3}: {100 * nonzero[c + 1:].sum() / max(n_nz, 1):.3f} %")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 500)
