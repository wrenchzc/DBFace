import numpy as np
import torch
import cv2
from torch.nn import functional as F
from typing import List

import common
import typing

from model.DBFace import DBFace

HAS_CUDA = torch.cuda.is_available()
print(f"HAS_CUDA = {HAS_CUDA}")

_model_cache = {}

def nms(objs, iou=0.5):
    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep


def detect(model, image, threshold=0.4, nms_iou=0.5) -> typing.List[common.BBox]:
    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]

    image = common.pad(image)
    image = ((image / 255.0 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)

    torch_image = torch.from_numpy(image)[None]
    if HAS_CUDA:
        torch_image = torch_image.cuda()

    hm, box, landmark = model(torch_image)
    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
    hm_height, hm_width = hm.shape[2:]

    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list((indices / hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
    scores = list(scores.data.numpy())
    box = box.cpu().squeeze().data.numpy()
    landmark = landmark.cpu().squeeze().data.numpy()

    stride = 4
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break

        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[:, cy, cx]
        x5y5 = (common.exp(x5y5 * 4) + ([cx] * 5 + [cy] * 5)) * stride
        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append(common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
    return nms(objs, iou=nms_iou)


def _get_resize_rate(size) -> float:
    MAX_LONG_SIDE = 1600

    h, w = size

    max_side = max(h, w)
    if max_side > MAX_LONG_SIDE:
        return MAX_LONG_SIDE / max_side
    else:
        return 1


def detect_image(model, file):
    raw_image = common.imread(file)
    if raw_image is None:
        raise ValueError(f"{file} is not a image file")
    return detect_image_by_nparray(model, raw_image)


def detect_image_by_nparray(model, raw_image: np.array) -> List[common.BBox]:
    scale_rate = _get_resize_rate(raw_image.shape[:2])
    image = cv2.resize(raw_image, (0, 0), fx=scale_rate, fy=scale_rate, interpolation=cv2.INTER_LINEAR)

    try:
        bboxes = detect(model, image)
        for bbox in bboxes:
            print(f"before rescale , {bbox}")
            bbox.rescale(scale_rate)
            print(f"after rescale , {bbox}")

        return bboxes
    except RuntimeError:  # face too big
        print("run time error, out of memory")
        raise


def _get_model():
    if "model" in _model_cache:
        return _model_cache["model"]

    _dbface = DBFace()
    _dbface.eval()

    if HAS_CUDA:
        _dbface.cuda()

    _dbface.load("model/dbface.pth")
    _model_cache["model"] = _dbface
    return _dbface