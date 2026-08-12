import cv2
import numpy as np
import math
from dataloader.eventloader import EventDataLoader

from einops import rearrange, reduce

ev = EventDataLoader("/scratch/sheusinger/st_stephan_360_640_20/train/")


def ev_repr_to_img(x: np.ndarray):
    ch, ht, wd = x.shape[-3:]
    assert ch > 1 and ch % 2 == 0
    ev_repr_reshaped = rearrange(x, "(posneg C) H W -> posneg C H W", posneg=2)
    img_neg = np.asarray(
        reduce(ev_repr_reshaped[0], "C H W -> H W", "sum"), dtype="int32"
    )
    img_pos = np.asarray(
        reduce(ev_repr_reshaped[1], "C H W -> H W", "sum"), dtype="int32"
    )
    img_diff = img_pos - img_neg
    # print(f"{img_diff.shape=}")
    img = 127 * np.ones((3, ht, wd), dtype=np.uint8)
    # print(f"{img.shape=}")
    img[:, img_diff > 0] = 255
    img[:, img_diff < 0] = 0
    return img


labels = ev.__getitem__(0)
img = np.load(
    "/scratch/sheusinger/st_stephan_360_640_20/train/2024_01_10_112814_drone_000/events/event_2.npz"
)["arr_0"]
print(img.shape)
img = ev_repr_to_img(img)
print(img.shape)
h = img.shape[1]
w = img.shape[2]
bs = 16
ns = np.ceil(bs**0.5)

img = np.transpose(img, (1, 2, 0))
mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
for i in range(bs):
    x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
    mosaic[y : y + h, x : x + w, :] = img

# scale = 1 / 16
max_size = 1920
scale = max_size / ns / max(h, w)


h = math.ceil(scale * h)
w = math.ceil(scale * w)
mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))


cv2.imshow("test", mosaic)
cv2.waitKey(0)
