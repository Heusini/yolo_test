import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from datasets.basedataset import create_matching_items, load_label
from multiprocessing.pool import ThreadPool
from itertools import repeat
from ultralytics.data.utils import save_dataset_cache_file, load_dataset_cache_file
from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr


DATASET_CACHE_VERSION = "1.0.4"  # 1.0.4: sequence dirs sorted (Match.seq/pos)


class YoloConverter(Dataset):
    def __init__(self, path, dataset: Dataset):
        self.path = Path(path)
        assert self.path.is_dir()
        self.dataset = dataset

        self.im_width = None
        self.im_height = None

        self.match_list = create_matching_items(self.path)

        cache_dir = Path("./cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_path_str = str(self.path.resolve()).strip("/").replace("/", "_")
        cache_name = f"{safe_path_str}_{self.__class__.__name__}.cache"
        self.cache_path = cache_dir / cache_name

        self.label_cache = None
        self.labels = self.get_labels()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        data = self.dataset.__getitem__(index)

        bboxes = data["bboxes"]
        img = data["img"]
        img_h, img_w = img.shape[-2], img.shape[-1]

        bboxes_yolo = torch.empty_like(bboxes, dtype=torch.float32)

        bboxes_yolo[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2.0 / img_w
        bboxes_yolo[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2.0 / img_h
        bboxes_yolo[:, 2] = (bboxes[:, 2] - bboxes[:, 0]) / img_w
        bboxes_yolo[:, 3] = (bboxes[:, 3] - bboxes[:, 1]) / img_h

        bboxes = bboxes_yolo

        data["bboxes"] = bboxes

        return data

    def get_labels(self):
        if self.cache_labels is not None:
            try:
                cache, exists = (
                    load_dataset_cache_file(self.cache_path),
                    True,
                )  # attempt to load a *.cache file
                assert (
                    cache["version"] == DATASET_CACHE_VERSION
                )  # matches current version
            except (
                FileNotFoundError,
                AssertionError,
                AttributeError,
                ModuleNotFoundError,
            ):
                cache, exists = (
                    self.cache_labels(),
                    False,
                )  # run cache ops
            self.cache_labels = cache["labels"]

        return self.cache_labels

    def cache_labels(self) -> dict:
        path = self.cache_path
        x = {"labels": []}
        total = self.__len__()

        im_height, im_width = self.get_im_shape()
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=load_label,
                iterable=zip(self.match_list, repeat(im_height), repeat(im_width)),
            )
            pbar = TQDM(results, desc="Creating label list", total=total)
            for bboxes, cls, ev_path, im_height, im_width in pbar:
                x["labels"].append(
                    {
                        "bboxes": bboxes,
                        "cls": cls,
                        "img_path": ev_path,
                        "ori_shape": im_height,
                        "resized_shape": im_width,
                    }
                )
        if x["labels"]:
            save_dataset_cache_file("Info", path, x, DATASET_CACHE_VERSION)
        return x

    def convert_boxes(self, labels, im_height, im_width):
        if len(labels) == 0:
            return np.empty((0, 4))
        bboxes = []
        for label in labels:
            x_center = (label["x"] + label["w"] / 2) / im_width
            y_center = (label["y"] + label["h"] / 2) / im_height
            bbox = np.array(
                [x_center, y_center, label["w"] / im_width, label["h"] / im_height]
            )
            bboxes.append(bbox)

        bboxes = np.stack(bboxes)
        return bboxes

    def get_im_shape(self):
        if self.im_width is None:
            event = np.load(self.match_list[0].event_path)
            self.im_width = event.shape[-1]
            self.im_height = event.shape[-2]

        return self.im_height, self.im_width
