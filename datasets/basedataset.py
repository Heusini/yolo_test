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


class Match:
    def __init__(self, event_path: Path, label_path: Path, frame_path: Path, seq: int = 0, pos: int = 0, seq_len: int = 1):
        self.event_path = event_path
        self.label_path = label_path
        self.frame_path = frame_path
        self.seq = seq  # index of the sequence directory
        self.pos = pos  # frame index inside the sequence
        self.seq_len = seq_len


def create_matching_items(path: Path):
    event_folder = Path("events")
    label_folder = Path("labels")
    frame_folder = Path("rgbs")
    match_list = []
    for seq, dir in enumerate(sorted(os.listdir(path))):
        event_path = path / dir / event_folder
        label_path = path / dir / label_folder
        rgb_path = path / dir / frame_folder

        event_files = os.listdir(event_path)
        label_files = os.listdir(label_path)
        label_files = [f for f in label_files if f.endswith(".npy")]
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
                seq=seq,
                pos=i,
                seq_len=len(event_files),
            )
            for i in range(len(event_files))
        ]
        match_list.extend(tmp_list)
        # return match_list

    return match_list


def load_label(args: tuple) -> list:
    match_list, img_height, img_width = args
    label = np.load(match_list.label_path)
    bboxes = convert_boxes(label, img_height, img_width)
    cls = label["class_id"]

    return [bboxes, cls, match_list.event_path, img_height, img_width]


def get_boxes_from_labels_xyxy(labels):
    if len(labels) == 0:
        return np.empty((0, 4))
    x, y, w, h = labels["x"], labels["y"], labels["w"], labels["h"]
    return np.stack([x, y, x + w, y + h], axis=-1)


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
    def __init__(self, path: str):
        self.path = Path(path)
        assert self.path.is_dir()
        self.match_list = create_matching_items(self.path)
        self.im_height, self.im_width = None, None

    def __len__(self):
        return len(self.match_list)
        # return 10
