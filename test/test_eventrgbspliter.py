import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.eventrgbdatasetspliter import EventRGBDatasetSpliter


eventsplitter = EventRGBDatasetSpliter(
    "/scratch/sheusinger/st_stephan_720_1280_10_10000/train/"
)


def box_to_xyxy(box, size):
    width = box[2] * size[1]
    height = box[3] * size[0]
    x1 = box[0] * size[1] - width / 2
    y1 = box[1] * size[0] - height / 2
    x2 = x1 + width
    y2 = y1 + height

    return np.array([x1, y1, x2, y2])


for item in eventsplitter:
    img = item["img"][:3, :, :].permute(1, 2, 0).numpy().copy()
    bboxes = item["bboxes"].numpy()
    for i, box in enumerate(bboxes):
        box = box_to_xyxy(box, item["ori_shape"][-2:])
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        img = cv2.rectangle(img, pt1, pt2, (255, 0, 0), 1)

    cv2.imshow("window", img)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
