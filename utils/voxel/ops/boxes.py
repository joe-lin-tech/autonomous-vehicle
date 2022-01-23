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


# TODO implementation of bev_iou
def box_iou(boxes1: Tensor, boxes2: Tensor):
    """
    Return birds-eye-view intersection-over-union between two sets of boxes.

    Args:
        boxes1 (Tensor[N, 9]): first set of boxes
        boxes2 (Tensor[M, 9]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise GIoU values for every element in boxes1 and boxes2
    """
    print("BOX IOU: ", boxes1.shape, boxes2.shape)
    boxes1_x1, boxes1_y1, boxes1_x2, boxes1_y2 = boxes1[:, 0:1] - boxes1[:, 3:4] / 2, boxes1[:, 1:2] - boxes1[:, 4:5] / 2, \
        boxes1[:, 0:1] + boxes1[:, 3:4] / 2, boxes1[:, 1:2] + boxes1[:, 4:5] / 2
    boxes2_x1, boxes2_y1, boxes2_x2, boxes2_y2 = boxes2[:, 0:1] - boxes2[:, 3:4] / 2, boxes2[:, 1:2] - boxes2[:, 4:5] / 2, \
        boxes2[:, 0:1] + boxes2[:, 3:4] / 2, boxes2[:, 1:2] + boxes2[:, 4:5] / 2
    boxes1 = torch.hstack((boxes1_x1, boxes1_y1, boxes1_x2, boxes1_y2))
    boxes2 = torch.hstack((boxes2_x1, boxes2_y1, boxes2_x2, boxes2_y2))
    print("AFTER TRANSFORM: ", boxes1.shape, boxes2.shape)
    return torchvision.ops.box_iou(boxes1, boxes2)

# TODO check implementation - adapted from original box_iou in torchvision


def box_giou(box1: Tensor, box2: Tensor) -> Tensor:
    """
    Return 3D generalized intersection-over-union between two 3D boxes.

    Args:
        box1 (Tensor[1, 9]): first box
        box2 (Tensor[1, 9]): second box

    Returns:
        float: the iou value of the two boxes
    """
    # inter, union = _box_inter_union(boxes1, boxes2)
    # iou = inter / union
    box1 = bbox.BBox3D(box1[0], box1[1], box1[2], length=box1[3], width=box1[4],
                       height=box1[5], euler_angles=[box1[6], box1[7], box1[8]])
    box2 = bbox.BBox3D(box2[0], box2[1], box2[2], length=box2[3], width=box2[4],
                       height=box2[5], euler_angles=[box2[6], box2[7], box2[8]])
    # iou = iou_3d(box1, box2)
    intersection_vol, iou = box3d_overlap(torch.as_tensor(
        box1.p[np.newaxis, :], dtype=torch.float), torch.as_tensor(box2.p[np.newaxis, :], dtype=torch.float))
    return iou

# TODO check functionality
# bounding box = (x, y, z, d, w, h, x_rot, y_rot, z_rot)
# x, y, z: location
# d, w, h: dimensions (depth - x, width - y, height - z)
# x_rot, y_rot, z_rot: rotation around x, y, z axis

# boxes - bounding boxes in a single frame [9, D, W]
# scores - scores for each bounding box in a single frame [1, D, W]


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
            if box_giou(curr_box, box) > iou_threshold:
                scores = torch.cat((scores[0:b, :], scores[b + 1:, :]), 0)
                boxes = torch.cat((boxes[0:b, :], boxes[b + 1:, :]), 0)

    # print("KEEP MASK: ", keep_mask.shape)
    # return keep_mask
    return preserved_boxes, preserved_scores


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    ds, ws, hs = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    keep = (ds >= min_size) & (ws >= min_size) & (hs >= min_size)
    keep = torch.where(keep)[0]
    return keep
