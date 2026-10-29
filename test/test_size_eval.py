"""Unit checks of the size-bucket filtering in test/size_eval.py (synthetic boxes, no model)."""

import sys

import torch

sys.path[:0] = ["./", "./test"]
from size_eval import bucket_filter, bucket_name, side


def box(x, y, s):
    return [x, y, x + s, y + s]


def test_side():
    assert torch.allclose(side(torch.tensor([[0, 0, 4, 9.0]])), torch.tensor([6.0]))


def test_gt_split_by_size():
    gt = torch.tensor([box(0, 0, 10), box(100, 100, 20), box(300, 300, 60)], dtype=torch.float)
    pred = torch.zeros(0, 4)
    gk, _ = bucket_filter(gt, pred, 0, 12)
    assert gk.tolist() == [True, False, False]
    gk, _ = bucket_filter(gt, pred, 12, 24)
    assert gk.tolist() == [False, True, False]
    gk, _ = bucket_filter(gt, pred, 48, float("inf"))
    assert gk.tolist() == [False, False, True]


def test_pred_matched_to_other_bucket_is_ignored():
    gt = torch.tensor([box(0, 0, 10), box(100, 100, 20)], dtype=torch.float)
    pred = torch.tensor([box(0, 0, 10), box(101, 101, 20)], dtype=torch.float)  # one hit per gt
    gk, pk = bucket_filter(gt, pred, 0, 12)
    assert gk.tolist() == [True, False] and pk.tolist() == [True, False]
    gk, pk = bucket_filter(gt, pred, 12, 24)
    assert gk.tolist() == [False, True] and pk.tolist() == [False, True]


def test_unmatched_pred_kept_only_in_its_own_bucket():
    gt = torch.tensor([box(0, 0, 10)], dtype=torch.float)
    pred = torch.tensor([box(500, 500, 10), box(600, 600, 30)], dtype=torch.float)  # two false positives
    _, pk = bucket_filter(gt, pred, 0, 12)
    assert pk.tolist() == [True, False]
    _, pk = bucket_filter(gt, pred, 24, 48)
    assert pk.tolist() == [False, True]


def test_pred_hitting_in_bucket_gt_kept_even_if_its_size_is_off():
    gt = torch.tensor([box(0, 0, 11)], dtype=torch.float)
    pred = torch.tensor([box(0, 0, 14)], dtype=torch.float)  # IoU 121/196 = 0.62, pred side 14 is outside <12
    gk, pk = bucket_filter(gt, pred, 0, 12)
    assert gk.tolist() == [True] and pk.tolist() == [True]


def test_empty_inputs():
    gk, pk = bucket_filter(torch.zeros(0, 4), torch.zeros(0, 4), 0, 12)
    assert gk.numel() == 0 and pk.numel() == 0
    gk, pk = bucket_filter(torch.zeros(0, 4), torch.tensor([box(0, 0, 10.0)]), 0, 12)
    assert pk.tolist() == [True]


def test_bucket_name():
    assert bucket_name(0, 12) == "<12 px" and bucket_name(12, 24) == "12-24 px" and bucket_name(48, float("inf")) == ">=48 px"


def test_update_metrics_accumulates_per_bucket():
    """Two images; a perfect 10 px and a perfect 30 px detection plus one 10 px false positive."""
    import numpy as np
    from size_eval import SizeValidator, bucket_results

    SizeValidator.setup([12, 24, 48])
    v = SizeValidator(args={"task": "detect", "plots": False, "save_json": False, "save_txt": False, "single_cls": False})
    v.device = torch.device("cpu")
    v.metrics.names = {0: "drone"}
    v.seen = 0
    H, W = 736, 1280
    gt = torch.tensor([box(100, 100, 10), box(500, 300, 30)], dtype=torch.float)  # image 0, image 1
    xywh = torch.stack([(gt[:, 0] + gt[:, 2]) / 2 / W, (gt[:, 1] + gt[:, 3]) / 2 / H, (gt[:, 2] - gt[:, 0]) / W, (gt[:, 3] - gt[:, 1]) / H], 1)
    batch = {
        "img": torch.zeros(2, 13, H, W), "batch_idx": torch.tensor([0.0, 1.0]), "cls": torch.zeros(2, 1), "bboxes": xywh,
        "ori_shape": [(720, 1280)] * 2, "ratio_pad": [((1.0, 1.0), (0.0, 0.0))] * 2, "im_file": ["a", "b"], "image_id": [0, 1],
    }
    preds = [
        {"bboxes": torch.tensor([box(100, 100, 10), box(900, 600, 10)], dtype=torch.float), "conf": torch.tensor([0.9, 0.5]), "cls": torch.zeros(2)},
        {"bboxes": torch.tensor([box(500, 300, 30)], dtype=torch.float), "conf": torch.tensor([0.8]), "cls": torch.zeros(1)},
    ]
    v.update_metrics(preds, batch)
    s = SizeValidator.size_stats
    small, mid, large = s[(0.0, 12.0)], s[(24.0, 48.0)], s[(48.0, float("inf"))]
    assert np.concatenate(small["target_cls"]).size == 1 and np.allclose(np.concatenate(small["conf"]), [0.9, 0.5])
    assert np.concatenate(small["tp"])[:, 0].tolist() == [True, False]
    assert np.concatenate(mid["target_cls"]).size == 1 and np.allclose(np.concatenate(mid["conf"]), [0.8])
    assert np.concatenate(mid["tp"])[0, -1]  # perfect box: TP at IoU 0.95 too
    assert np.concatenate(large["target_cls"]).size == 0 and np.concatenate(large["conf"]).size == 0
    n_gt, r, m50, m = bucket_results(mid)
    assert n_gt == 1 and r > 0.99 and m50 > 0.99 and m > 0.99
    n_gt, r, m50, m = bucket_results(small)
    assert n_gt == 1 and r > 0.99 and m50 > 0.99  # the false positive ranks below the true positive
    all_stats = {k: np.concatenate(v_, 0) for k, v_ in v.metrics.stats.items()}
    assert all_stats["target_cls"].size == 2 and all_stats["conf"].size == 3
