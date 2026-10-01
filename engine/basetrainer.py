import torch


def collate_fn(batch):
    imgs = torch.stack([b["img"] for b in batch], 0)
    ori_shapes = torch.stack([torch.tensor(b["ori_shape"]) for b in batch], 0)
    resized_shapes = torch.stack([torch.tensor(b["resized_shape"]) for b in batch], 0)
    image_ids = torch.stack([torch.tensor(b["image_id"]) for b in batch], 0)
    cls = torch.vstack([b["cls"] for b in batch])
    im_files = [b["im_file"] for b in batch]
    bboxes = torch.vstack([b["bboxes"] for b in batch])
    batch_idx = []
    for i, b in enumerate(batch):
        n = b["cls"].shape[0]
        batch_idx.append(torch.full((n,), i, dtype=torch.long))
    batch_idx = torch.cat(batch_idx)
    return {
        "img": imgs,
        "cls": cls,
        "bboxes": bboxes,
        "batch_idx": batch_idx,
        "im_file": im_files,
        "ori_shape": ori_shapes,
        "resized_shape": resized_shapes,
        "ratio_pad": [((1.0, 1.0), (0.0, 0.0))] * len(imgs),
        "image_id": image_ids,
    }
