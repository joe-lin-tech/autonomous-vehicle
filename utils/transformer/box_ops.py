import torch
import torchvision
import bbox
import numpy as np
from pytorch3d.ops import box3d_overlap


def bev_iou(boxes1, boxes2):
    """
    Return intersection-over-union of boxes1 and boxes2 in BEV.

    Args:
        boxes1 (Tensor[N, 9]): first set of boxes
        boxes2 (Tensor[M, 9]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise BEV IOU values for every element in boxes1 and boxes2
    """
    boxes1_x1, boxes1_y1, boxes1_x2, boxes1_y2 = boxes1[:, 0:1] - boxes1[:, 3:4] / 2, boxes1[:, 1:2] - boxes1[:, 4:5] / 2, \
        boxes1[:, 0:1] + boxes1[:, 3:4] / \
        2, boxes1[:, 1:2] + boxes1[:, 4:5] / 2
    boxes2_x1, boxes2_y1, boxes2_x2, boxes2_y2 = boxes2[:, 0:1] - boxes2[:, 3:4] / 2, boxes2[:, 1:2] - boxes2[:, 4:5] / 2, \
        boxes2[:, 0:1] + boxes2[:, 3:4] / \
        2, boxes2[:, 1:2] + boxes2[:, 4:5] / 2
    boxes1 = torch.hstack((boxes1_x1, boxes1_y1, boxes1_x2, boxes1_y2))
    boxes2 = torch.hstack((boxes2_x1, boxes2_y1, boxes2_x2, boxes2_y2))
    return torchvision.ops.box_iou(boxes1, boxes2)


def generalized_iou(box1, box2):
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


def batched_nms(boxes, scores, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.

    Args:
        boxes (Tensor[N, 9]): boxes where NMS will be performed
        scores (Tensor[N]): scores for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: boxes that have been kept by NMS, sorted in decreasing order of scores
        Tensor: the scores of the elements that have been kept by NMS, sorted in decreasing order.
    """
    preserved_boxes, preserved_scores = [], []
    while boxes.shape[0] > 0:
        max_index = torch.argmax(scores)
        curr_box = boxes[max_index]
        preserved_boxes.append(curr_box)
        preserved_scores.append(scores[max_index])
        scores = torch.cat(
            (scores[0:max_index, :], scores[max_index + 1:, :]), 0)
        boxes = torch.cat((boxes[0:max_index, :], boxes[max_index + 1:, :]), 0)
        for b, box in enumerate(boxes.numpy()):
            if generalized_iou(curr_box, box) > iou_threshold:
                scores = torch.cat((scores[0:b, :], scores[b + 1:, :]), 0)
                boxes = torch.cat((boxes[0:b, :], boxes[b + 1:, :]), 0)
    return preserved_boxes, preserved_scores
