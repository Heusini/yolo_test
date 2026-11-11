"""Build a mixed dataset folder out of symlinks: SkyEV + FRED train sequences, SkyEV val only.

    python utils/make_mixed_dataset.py <out dir> <skyev root> <fred root> [--fred-every N] [--fred-val]

<out>/train/<seq>       -> <skyev>/train/<seq>
<out>/train/fred_<id>   -> <fred>/train/<id>      (every N-th FRED sequence with --fred-every N)
<out>/val/<seq>         -> <skyev>/val/<seq>
<out>/val/fred_<id>     -> <fred>/val/<id>        (only with --fred-val; the default val is SkyEV only)
The loader reads npy and jpg/npz sequences side by side; FRED val is evaluated separately with test/robustness_eval.py.
"""

import argparse
import os
from pathlib import Path


def link_split(out: Path, src: Path, prefix: str = "", every: int = 1) -> int:
    out.mkdir(parents=True, exist_ok=True)
    names = sorted((d for d in os.listdir(src) if (src / d).is_dir()), key=lambda d: (len(d), d))[::every]
    for name in names:
        link = out / f"{prefix}{name}"
        if link.is_symlink():
            link.unlink()
        link.symlink_to((src / name).resolve())
    return len(names)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("skyev", type=Path)
    p.add_argument("fred", type=Path)
    p.add_argument("--fred-every", type=int, default=1, help="link every N-th FRED train sequence (1 = all)")
    p.add_argument("--fred-val", action="store_true", help="also link the FRED val sequences into val/")
    a = p.parse_args()

    n = {
        "train skyev": link_split(a.out / "train", a.skyev / "train"),
        "train fred": link_split(a.out / "train", a.fred / "train", "fred_", a.fred_every),
        "val skyev": link_split(a.out / "val", a.skyev / "val"),
    }
    if a.fred_val:
        n["val fred"] = link_split(a.out / "val", a.fred / "val", "fred_")
    for k, v in n.items():
        print(f"{k}: {v} sequences")


if __name__ == "__main__":
    main()
