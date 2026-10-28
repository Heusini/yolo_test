"""ZoomTransformer: shapes/dtypes kept, event counts stay integers (in: original values, out: mass kept),
boxes rescaled and clipped, p=0 is identity."""

import random
import sys

import torch

sys.path.append("./")
from datasets.aramsuisse_dataset import ArmasuisseDataset
from datasets.zoom_transformer import ZoomTransformer

root = "/home/sebastian/.cache/yolo_test_cpu/fake/val"
ds = ArmasuisseDataset(root, True, True)
img, boxes = ds[1]["img"], ds[1]["bboxes"]
h, w = img.shape[-2:]

assert torch.equal(ZoomTransformer(ds, p=0.0)[0]["img"], ds[0]["img"])

# zoom-in by 2 from (40, 100)
out, b, keep = ZoomTransformer.zoom_in(img, boxes, 40, 100, h // 2, w // 2, 3, 0.5)
assert out.shape == img.shape and out.dtype == img.dtype
assert set(torch.unique(out[3:]).tolist()) <= set(torch.unique(img[3:]).tolist())
_, b, keep = ZoomTransformer.zoom_in(img, torch.tensor([[150.0, 60.0, 170.0, 80.0]]), 40, 100, h // 2, w // 2, 3, 0.5)
assert torch.allclose(b, torch.tensor([[100.0, 40.0, 140.0, 80.0]])) and keep.all()
_, b, _ = ZoomTransformer.zoom_in(img, torch.tensor([[0.0, 0.0, 10.0, 10.0]]), 40, 100, h // 2, w // 2, 3, 0.5)
assert len(b) == 0  # outside the crop
_, b, _ = ZoomTransformer.zoom_in(img, torch.tensor([[90.0, 60.0, 110.0, 80.0]]), 40, 100, h // 2, w // 2, 3, 0.5)
assert len(b) == 1 and b[0, 0] == 0  # half visible, kept and clipped

# zoom-out by 2, pasted at (20, 30)
out, b, keep = ZoomTransformer.zoom_out(img, torch.tensor([[150.0, 60.0, 170.0, 80.0]]), 20, 30, h // 2, w // 2, 3, 0.5)
assert out.shape == img.shape and out.dtype == img.dtype
assert out[:, :20].sum() == 0 and out[:, :, :30].sum() == 0 and out[:, 20 + h // 2 :].sum() == 0  # zero canvas
assert torch.allclose(b, torch.tensor([[105.0, 50.0, 115.0, 60.0]])) and keep.all()
mass_in, mass_out = img[3:].float().sum(), out[3:].float().sum()
assert 0.9 * mass_in <= mass_out <= 1.05 * mass_in, (mass_in, mass_out)  # area-summed, clipped at the cutoff
assert out[3:].max() <= 10
# a 2x2 blob of single events survives the 2x shrink as one pixel with count 4
blob = torch.zeros_like(img)
blob[5, 100:102, 200:202] = 1
out, _, _ = ZoomTransformer.zoom_out(blob, torch.zeros(0, 4), 0, 0, h // 2, w // 2, 3, 0.5)
assert out[5].sum() == 4 and out[5].max() == 4

# through the wrapper: both branches run, cls follows the boxes
z = ZoomTransformer(ds, p=1.0, zin=1.5, zout=1.3)
random.seed(0)
for i in range(len(ds)):
    d = z[i]
    assert d["img"].shape == ds[i]["img"].shape and d["img"].dtype == ds[i]["img"].dtype
    assert len(d["cls"]) == len(d["bboxes"])
print("ok")
