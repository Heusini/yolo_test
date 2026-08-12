import json
import torch
from typing import Any, List, Tuple
import os
import numpy as np

import random

from torch.utils.data import Dataset
from pathlib import Path
from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr
from ultralytics.data.utils import save_dataset_cache_file, load_dataset_cache_file
from multiprocessing.pool import ThreadPool
from itertools import repeat


DATASET_CACHE_VERSION = "1.0.3"


class Match:
    def __init__(self, event_path: Path, label_path: Path, frame_path: Path):
        self.event_path = event_path
        self.label_path = label_path
        self.frame_path = frame_path


def create_matching_items(path: Path):
    event_folder = Path("events")
    label_folder = Path("labels")
    frame_folder = Path("rgbs")
    match_list = []
    for dir in os.listdir(path):
        event_path = path / dir / event_folder
        label_path = path / dir / label_folder
        rgb_path = path / dir / frame_folder

        event_files = os.listdir(event_path)
        label_files = os.listdir(label_path)
        label_files = [f for f in label_files if f.endswith(".npz")]
        rgb_files = os.listdir(rgb_path)

        event_files.sort(key=lambda item: (len(item), item))
        label_files.sort(key=lambda item: (len(item), item))
        rgb_files.sort(key=lambda item: (len(item), item))
        assert_msg = f"event_len({len(event_files)}) != label_len({len(label_files)}) != rgb_len({len(rgb_files)}) for\n{event_path},\n{label_path} and\n{rgb_path}"
        assert len(event_files) > 0, f"event_files empty, {event_path}"
        assert len(label_files) > 0, f"event_files empty, {label_path}"
        assert len(rgb_files) > 0, f"event_files empty, {rgb_path}"
        assert len(event_files) == len(label_files) == len(rgb_files), assert_msg

        tmp_list = [
            Match(
                event_path / event_files[i],
                label_path / label_files[i],
                rgb_path / rgb_files[i],
            )
            for i in range(len(event_files))
        ]
        match_list.extend(tmp_list)
        # return match_list

    return match_list


def load_label(args: tuple) -> list:
    match_list, img_height, img_width = args
    label = np.load(match_list.label_path)["arr_0"]
    bboxes = convert_boxes(label, img_height, img_width)
    cls = label["class_id"]

    return [bboxes, cls, match_list.event_path, img_height, img_width]


def convert_boxes(labels, im_height, im_width):
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


class BaseDataset(Dataset):
    def __init__(self, path: str, padding=(0, 0, 0, 0)):
        self.path = Path(path)
        self.padding = padding
        assert self.path.is_dir()
        self.match_list = create_matching_items(self.path)
        self.im_height, self.im_width = None, None
        self.im_height_padded, self.im_width_padded = None, None
        
        cache_dir = Path("./cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_path_str = str(self.path.resolve()).strip("/").replace("/", "_")
        cache_name = f"{safe_path_str}_{self.__class__.__name__}.cache"
        self.cache_path = cache_dir / cache_name
        
        self.labels = self.get_labels()

    def get_labels(self):
        try:
            cache, exists = (
                load_dataset_cache_file(self.cache_path),
                True,
            )  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
        except FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError:
            cache, exists = self.cache_labels(self.cache_path), False  # run cache ops

        labels = cache["labels"]
        return labels

    def cache_labels(self, path: Path = None) -> dict:
        path = path or self.cache_path
        x = {"labels": []}
        padded_size = self.get_im_padded_shape()
        total = self.__len__()
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=load_label,
                iterable=zip(
                    self.match_list, repeat(padded_size[0]), repeat(padded_size[1])
                ),
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

    def get_im_padded_shape(self):
        if self.im_width_padded is None:
            shape = self.get_im_shape()
            self.im_height_padded = shape[0] + sum(self.padding[2:])
            self.im_width_padded = shape[1] + sum(self.padding[:2])
        return self.im_height_padded, self.im_width_padded

    def get_im_shape(self):
        if self.im_width is None:
            event = np.load(self.match_list[0].event_path)["arr_0"]
            self.im_width = event.shape[-1]
            self.im_height = event.shape[-2]
        return self.im_height, self.im_width

    # def init_labels(self):
    #     label_list = []
    #     count = 0
    #     padded_size = self.get_im_padded_shape()
    #     for match in self.match_list:
    #         labels = np.load(match.label_path)["arr_0"]

    #         bboxes = self.convert_boxes(labels, padded_size[0], padded_size[1])
    #         cls = labels["class_id"]
    #         # unique_id = "/".join(match.event_path.parts[-3:])
    #         # print(unique_id)
    #         label = {
    #             "bboxes": bboxes,
    #             "cls": cls,
    #             "img_path": match.event_path,
    #             "ori_shape": padded_size,
    #             "resized_shape": padded_size,
    #         }
    #         count += 1
    #         label_list.append(label)
    #     return label_list

    def create_coco_annotatioins(self):
        images = []
        annotations = []
        categories = []
        drone_category = {"id": 1, "name": "drone"}
        categories.append(drone_category)

        event_shape = self.get_im_shape()
        box_count = len(self)
        for i, match in enumerate(self.match_list):
            event = {}
            event["id"] = i
            event["height"] = 384
            event["width"] = 640
            event["file_name"] = str(match.event_path)

            images.append(event)

            labels = np.load(match.label_path)["arr_0"]
            cls = labels["class_id"]
            for j, label in enumerate(labels):
                annotation = {}
                annotation["id"] = box_count
                annotation["image_id"] = i
                annotation["category_id"] = 1
                annotation["area"] = float(label["w"] * label["h"])
                annotation["bbox"] = [
                    float(label["x"]),
                    float(label["y"]),
                    float(label["w"]),
                    float(label["h"]),
                ]
                annotation["iscrowd"] = 0
                annotations.append(annotation)
                box_count += 1
        coco = {}
        coco["images"] = images
        coco["annotations"] = annotations
        coco["categories"] = categories

        json_str = json.dumps(coco)
        return json_str

    def __len__(self):
        return len(self.match_list)
        # return 10

    def check_cache_ram(self, safety_margin: float = 0.5) -> bool:
        """Check if there's enough RAM for caching images.

        Args:
            safety_margin (float): Safety margin factor for RAM calculation.

        Returns:
            (bool): True if there's enough RAM, False otherwise.
        """
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.__len__(), 30)  # extrapolate from 30 random images
        for _ in range(n):
            print(random.choice(self.match_list).event_path)
            im = np.load(random.choice(self.match_list).event_path)[
                "arr_0"
            ]  # sample image
            print(im.shape)
            if im is None:
                continue
            # ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            print(im.nbytes)
            b += im.nbytes
        mem_required = (
            b * self.__len__() / n * (1 + safety_margin)
        )  # GB required to cache dataset into RAM
        mem = __import__("psutil").virtual_memory()
        print(f"{mem_required / gb:.5f} GB RAM")
        if mem_required > mem.available:
            self.cache = None
            print(
                # LOGGER.warning(
                f"Error: {mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images"
            )
            return False
        return True
