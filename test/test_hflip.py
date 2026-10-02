"""CPU check of HFlipTransformer. usage (repo root): PYTHONPATH=. python test/test_hflip.py <fake_dataset_dir>"""

import sys

import torch

from datasets.aramsuisse_dataset import ArmasuisseDataset
from datasets.flip_transformer import HFlipTransformer


def main(root: str):
    base = ArmasuisseDataset(f"{root}/train", True, True)
    idx = next(i for i in range(len(base)) if len(base[i]["bboxes"]))
    x, boxes = base[idx]["img"], base[idx]["bboxes"]
    w = x.shape[-1]

    f = HFlipTransformer(base, p=1.0)[idx]
    assert torch.equal(f["img"], torch.flip(x, dims=[-1])), "all 13 channels must be mirrored"
    fb = f["bboxes"]
    assert torch.allclose(fb[:, 0], w - boxes[:, 2]) and torch.allclose(fb[:, 2], w - boxes[:, 0])
    assert torch.equal(fb[:, [1, 3]], boxes[:, [1, 3]]) and (fb[:, 2] > fb[:, 0]).all()
    # the box content is the mirrored content
    b, g = boxes[0].long(), fb[0].long()
    assert torch.equal(x[:, b[1]:b[3], b[0]:b[2]], torch.flip(f["img"][:, g[1]:g[3], g[0]:g[2]], dims=[-1]))
    assert torch.equal(HFlipTransformer(base, p=0.0)[idx]["img"], x)
    print(f"ok: image mirrored, boxes mirrored (x1'=w-x2), p=0 identity")


if __name__ == "__main__":
    main(sys.argv[1])
