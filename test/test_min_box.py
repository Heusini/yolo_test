"""MinBoxTransformer: small boxes grow around their center to `size`, big ones are untouched, growth is clipped
at the image border, size=0 is identity. usage (repo root): PYTHONPATH=. python test/test_min_box.py <fake_dataset_dir>"""

import sys

import torch

from datasets.aramsuisse_dataset import ArmasuisseDataset
from datasets.min_box_transformer import MinBoxTransformer

boxes = torch.tensor([[100.0, 50.0, 110.0, 56.0], [200.0, 100.0, 260.0, 140.0], [2.0, 3.0, 8.0, 7.0]])
out = MinBoxTransformer.grow(boxes, 24, 384, 640)
assert torch.allclose(out[0], torch.tensor([93.0, 41.0, 117.0, 65.0]))  # 10x6 -> 24x24 around (105, 53)
assert torch.equal(out[1], boxes[1])  # 60x40 stays
assert torch.allclose(out[2], torch.tensor([0.0, 0.0, 17.0, 17.0]))  # clipped at the top-left corner
assert torch.equal(MinBoxTransformer.grow(boxes, 0, 384, 640), boxes)
assert len(MinBoxTransformer.grow(torch.zeros(0, 4), 24, 384, 640)) == 0

root = sys.argv[1]
base = ArmasuisseDataset(f"{root}/train", True, True)
idx = next(i for i in range(len(base)) if len(base[i]["bboxes"]))
d = MinBoxTransformer(base, 32)[idx]
wh = d["bboxes"][:, 2:] - d["bboxes"][:, :2]
c0 = (base[idx]["bboxes"][:, :2] + base[idx]["bboxes"][:, 2:]) / 2
c1 = (d["bboxes"][:, :2] + d["bboxes"][:, 2:]) / 2
inside = (d["bboxes"][:, :2] > 0).all(1) & (d["bboxes"][:, 2] < d["img"].shape[-1]) & (d["bboxes"][:, 3] < d["img"].shape[-2])
assert torch.allclose(c0[inside], c1[inside]) and (wh[inside] >= 32).all()
assert torch.equal(d["img"], base[idx]["img"]) and torch.equal(d["cls"], base[idx]["cls"])
print(f"ok: {len(d['bboxes'])} boxes grown to >= 32 px, centers kept, image and cls untouched")
