import torch
from torch import Tensor
from typing import Tuple
from utils.voxel.ops._box_convert import _box_cxcywh_to_xyxy, _box_xyxy_to_cxcywh, _box_xywh_to_xyxy, _box_xyxy_to_xywh
import torchvision
from torchvision.extension import _assert_has_ops
import numpy as np
import bbox
# from bbox.metrics import iou_3d

# TODO revert torch version back to latest (pytorch3d only works for torch 1.9.0)
from pytorch3d.ops import box3d_overlap


# TODO implement 3D-GIoU, update to input Tensors not np.arrays
def iou(box_a: Tensor, box_b: Tensor):
    box_a = bbox.BBox3D(box_a[0], box_a[1], box_a[2], length=box_a[3], width=box_a[4],
                        height=box_a[5], euler_angles=[box_a[6], box_a[7], box_a[8]])
    box_b = bbox.BBox3D(box_b[0], box_b[1], box_b[2], length=box_b[3], width=box_b[4],
                        height=box_b[5], euler_angles=[box_b[6], box_b[7], box_b[8]])
    # iou = iou_3d(box_a, box_b)
    intersection_vol, iou = box3d_overlap(torch.as_tensor(
        box_a.p[np.newaxis, :], dtype=torch.float), torch.as_tensor(box_b.p[np.newaxis, :], dtype=torch.float))
    return iou

# TODO check implementation - adapted from original box_iou in torchvision

# TODO update to input tensors instead of np.arrays


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union between two sets of boxes.

    Args:
        boxes1 (Tensor[N, 9]): first set of boxes
        boxes2 (Tensor[M, 9]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    # inter, union = _box_inter_union(boxes1, boxes2)
    # iou = inter / union
    ious = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            ious[i, j] = iou(box1, box2)
    return torch.as_tensor(ious)

# TODO check functionality
# bounding box = (x, y, z, d, w, h, x_rot, y_rot, z_rot)
# x, y, z: location
# d, w, h: dimensions (depth - x, width - y, height - z)
# x_rot, y_rot, z_rot: rotation around x, y, z axis

# boxes - bounding boxes in a single frame [9, D, W]
# scores - scores for each bounding box in a single frame [1, D, W]


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    Performs non-maximum suppression given boxes and scores.

    Args:
        boxes - bounding boxes in a single frame [D * W, 9]
        scores - scores for each bounding box in a single frame [D * W, 1]
    """
    print("NMS: ", boxes.shape, scores.shape)
    preserved_boxes = []
    preserved_scores = []
    while boxes.shape[0] > 0:
        max_index = torch.argmax(scores)
        curr_box = boxes[max_index]
        preserved_boxes.append(curr_box)
        preserved_scores.append(scores[max_index])
        scores = torch.cat(
            (scores[0:max_index, :], scores[max_index + 1:, :]), 0)
        boxes = torch.cat((boxes[0:max_index, :], boxes[max_index + 1:, :]), 0)
        for b, box in enumerate(boxes.numpy()):
            if iou(curr_box, box) > iou_threshold:
                scores = torch.cat((scores[0:b, :], scores[b + 1:, :]), 0)
                boxes = torch.cat((boxes[0:b, :], boxes[b + 1:, :]), 0)

    # print("KEEP MASK: ", keep_mask.shape)
    # return keep_mask
    return preserved_boxes, preserved_scores


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    # idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 9]): boxes where NMS will be performed.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    # Benchmarks that drove the following thresholds are at
    # https://github.com/pytorch/vision/issues/1311#issuecomment-781329339
    # Ideally for GPU we'd use a higher threshold
    return nms(boxes, scores, iou_threshold)


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    ds, ws, hs = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    keep = (ds >= min_size) & (ws >= min_size) & (hs >= min_size)
    keep = torch.where(keep)[0]
    return keep