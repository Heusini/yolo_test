"""Per-size mAP of a checkpoint: the validation metrics split by ground-truth box size (no training).

usage (repo root):
  python test/size_eval.py <ckpt.pt> [<ckpt2.pt> ...] --data conf/<data>.yaml --imgsz 1280 --device 0
  optional: --edges 12 24 48   (bucket borders in px of sqrt(w*h), default; gives <12, 12-24, 24-48, >=48)
            --condition clean  (any row of test/robustness_eval.py, e.g. rgb_zero)
            --coco             (additionally the official pycocotools numbers: AP, AP50, AP75, AP_S/M/L, AR; for
                                comparison with published models; COCO small = area < 32^2 px, i.e. --edges 32 96)

COCO-style size evaluation: for a bucket, ground-truth boxes outside the bucket are ignored, predictions that match an
ignored box (IoU >= 0.5) are dropped, and unmatched predictions whose own size lies outside the bucket are dropped
too, so a bucket is neither punished for finding the other sizes nor for missing them. Sizes are measured in input
pixels (the 720p export is fed at native resolution). The "all" row is the normal validation result.

The bucket rows use the Ultralytics AP (all-point interpolation, up to 300 detections per image); --coco runs
pycocotools on the same predictions (101-point interpolation, 100 detections), which is what papers report.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = ["./", "./test"]
from datasets.aramsuisse_dataset import ArmasuisseDataset
from robustness_eval import CONDITIONS, RobustValidator
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.metrics import ap_per_class, box_iou

MATCH_IOU = 0.5


def side(boxes: torch.Tensor) -> torch.Tensor:
    """xyxy boxes -> sqrt(w * h)."""
    return ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clamp(min=0).sqrt()


def bucket_filter(gt: torch.Tensor, pred: torch.Tensor, lo: float, hi: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean masks (gt_keep, pred_keep) for the size bucket [lo, hi)."""
    gt_keep = (side(gt) >= lo) & (side(gt) < hi)
    pred_side = side(pred)
    pred_keep = (pred_side >= lo) & (pred_side < hi)
    if len(gt) and len(pred):
        iou = box_iou(gt, pred)  # [M, N]
        hit_in = (iou[gt_keep] >= MATCH_IOU).any(0) if gt_keep.any() else torch.zeros(len(pred), dtype=torch.bool, device=pred.device)
        hit_out = (iou[~gt_keep] >= MATCH_IOU).any(0) if (~gt_keep).any() else torch.zeros_like(hit_in)
        pred_keep = hit_in | (pred_keep & ~hit_out)
    return gt_keep, pred_keep


class SizeValidator(RobustValidator):
    edges = (12.0, 24.0, 48.0)
    buckets: list[tuple[float, float]] = []
    size_stats: dict = {}  # bucket -> per-image stats (class attribute: survives model.val(); BaseValidator owns self.stats)
    coco_preds: list | None = None  # COCO result dicts when --coco is set

    @classmethod
    def setup(cls, edges, coco=False):
        cls.edges = tuple(edges)
        e = (0.0, *cls.edges, float("inf"))
        cls.buckets = list(zip(e[:-1], e[1:]))
        cls.size_stats = {b: {k: [] for k in ("tp", "conf", "pred_cls", "target_cls")} for b in cls.buckets}
        cls.coco_preds = [] if coco else None

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for si, pred in enumerate(preds):
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)
            if self.coco_preds is not None:  # input pixels == original pixels (bottom pad only, ratio 1)
                self.coco_preds.extend(coco_results(int(pbatch["image_id"]), predn["bboxes"], predn["conf"], predn["cls"]))
            for b in self.buckets:
                gk, pk = bucket_filter(pbatch["bboxes"], predn["bboxes"], *b)
                sub_gt = {"cls": pbatch["cls"][gk], "bboxes": pbatch["bboxes"][gk]}
                sub_pred = {k: v[pk] for k, v in predn.items() if isinstance(v, torch.Tensor) and v.shape[:1] == pk.shape}
                st = self.size_stats[b]
                st["tp"].append(self._process_batch(sub_pred, sub_gt)["tp"])
                st["conf"].append(sub_pred["conf"].cpu().numpy())
                st["pred_cls"].append(sub_pred["cls"].cpu().numpy())
                st["target_cls"].append(sub_gt["cls"].cpu().numpy())


def coco_results(image_id: int, xyxy: torch.Tensor, conf: torch.Tensor, cls: torch.Tensor) -> list[dict]:
    """Predictions of one image as COCO result dicts (bbox = x, y, w, h in pixels, category_id = class + 1)."""
    b = xyxy.cpu().numpy()
    return [{"image_id": image_id, "category_id": int(c) + 1, "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(s)}
            for (x1, y1, x2, y2), s, c in zip(b, conf.cpu().numpy(), cls.cpu().numpy())]


def coco_ground_truth(val_dir: str, names: dict) -> dict:
    """COCO ground-truth dict of a split; image_id = dataset index, same order as the validator's loader."""
    ds = ArmasuisseDataset(val_dir, True, False)
    h, w = ds.get_im_shape()
    images, anns = [], []
    for i, m in enumerate(ds.match_list):
        images.append({"id": i, "file_name": str(m.label_path), "height": int(h), "width": int(w)})
        for lab in np.load(m.label_path):
            bw, bh = float(lab["w"]), float(lab["h"])
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": int(lab["class_id"]) + 1,
                         "bbox": [float(lab["x"]), float(lab["y"]), bw, bh], "area": bw * bh, "iscrowd": 0})
    cats = [{"id": int(k) + 1, "name": v} for k, v in names.items()]
    return {"images": images, "annotations": anns, "categories": cats}


COCO_KEYS = ["AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L", "AR1", "AR10", "AR100", "AR_S", "AR_M", "AR_L"]


def coco_eval(gt: dict, preds: list[dict]) -> dict[str, float]:
    """pycocotools bbox evaluation -> the 12 standard numbers (nan if there are no predictions)."""
    if not preds:
        return dict.fromkeys(COCO_KEYS, float("nan"))
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        c_gt = COCO()
        c_gt.dataset = gt
        c_gt.createIndex()
        c_dt = c_gt.loadRes(preds)
        ev = COCOeval(c_gt, c_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return dict(zip(COCO_KEYS, ev.stats.tolist()))


def bucket_results(st: dict) -> tuple[int, float, float, float]:
    """-> (n_gt, recall at max F1, mAP50, mAP50-95) of one bucket."""
    s = {k: np.concatenate(v, 0) for k, v in st.items()}
    n_gt = len(s["target_cls"])
    if n_gt == 0 or len(s["conf"]) == 0:
        return n_gt, float("nan"), float("nan"), float("nan")
    _, _, _, r, _, ap, *_ = ap_per_class(s["tp"], s["conf"], s["pred_cls"], s["target_cls"], plot=False)
    return n_gt, float(r.mean()), float(ap[:, 0].mean()), float(ap.mean())


def bucket_name(lo, hi):
    return f"<{hi:g} px" if lo == 0 else (f">={lo:g} px" if hi == float("inf") else f"{lo:g}-{hi:g} px")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--edges", type=float, nargs="+", default=[12, 24, 48])
    p.add_argument("--condition", default="clean", choices=CONDITIONS)
    p.add_argument("--clip-len", type=int, default=1)
    p.add_argument("--coco", action="store_true", help="also report the pycocotools numbers (AP_S = area < 32^2)")
    a = p.parse_args()
    SizeValidator.CLIP_LEN = a.clip_len
    SizeValidator.condition = a.condition

    torch.manual_seed(0)
    rows, coco = {}, {}
    data = check_det_dataset(a.data)
    gt = coco_ground_truth(data["val"], data["names"]) if a.coco else None
    for ckpt in a.ckpts:
        name = Path(ckpt).resolve().parent.parent.name
        SizeValidator.setup(a.edges, coco=a.coco)
        m = YOLO(ckpt).val(data=a.data, imgsz=a.imgsz, device=a.device, batch=a.batch, workers=a.workers, rect=True,
                           plots=False, verbose=False, validator=SizeValidator, project="runs/size_eval", name=f"{name}_{a.condition}", exist_ok=True)
        rows[(name, "all")] = (int(m.nt_per_class.sum()), m.box.mr, m.box.map50, m.box.map)
        for b in SizeValidator.buckets:
            rows[(name, bucket_name(*b))] = bucket_results(SizeValidator.size_stats[b])
        if a.coco:
            coco[name] = coco_eval(gt, SizeValidator.coco_preds)

    print(f"\n=== per-size results ({a.condition}) ===")
    names = list(dict.fromkeys(n for n, _ in rows))
    labels = ["all"] + [bucket_name(*b) for b in SizeValidator.buckets]
    for n in names:
        print(f"\n{n}")
        print(f"{'bucket':12s}{'n_gt':>8s}{'recall':>9s}{'mAP50':>9s}{'mAP50-95':>10s}")
        for lab in labels:
            n_gt, r, m50, m = rows[(n, lab)]
            print(f"{lab:12s}{n_gt:8d}{r:9.3f}{m50:9.3f}{m:10.3f}")
        if a.coco:
            print("pycocotools: " + "  ".join(f"{k} {100 * coco[n][k]:.1f}" for k in COCO_KEYS[:6]) + "  |  " + "  ".join(f"{k} {100 * coco[n][k]:.1f}" for k in COCO_KEYS[6:]))


if __name__ == "__main__":
    main()
