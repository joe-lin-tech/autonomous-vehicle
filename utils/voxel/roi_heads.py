from typing import Optional, List, Dict, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from utils.voxel.ops import boxes as box_ops
from torchvision.ops import roi_align
from data_types.target import VoxelTarget

from utils._utils import *


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.
    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    print("LABELS: ", len(labels), labels[0].shape)
    print("REGRESSION TARGETS: ", len(regression_targets), regression_targets[0].shape)
    print("CLASS LOGITS: ", class_logits.shape)

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    class_logits = torch.cat([class_logits[0], class_logits[1]], dim=0)

    print("LABELS: ", len(labels), labels[0].shape)
    print("REGRESSION TARGETS: ", len(regression_targets), regression_targets[0].shape)
    print("CLASS LOGITS: ", class_logits.shape)
    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 9, 9)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


# def maskrcnn_inference(x, labels):
#     # type: (Tensor, List[Tensor]) -> List[Tensor]
#     """
#     From the results of the CNN, post process the masks
#     by taking the mask corresponding to the class with max
#     probability (which are of fixed size and directly output
#     by the CNN) and return the masks in the mask field of the BoxList.
#     Args:
#         x (Tensor): the mask logits
#         labels (list[BoxList]): bounding boxes that are used as
#             reference, one for ech frame
#     Returns:
#         results (list[BoxList]): one BoxList for each frame, containing
#             the extra field mask
#     """
#     mask_prob = x.sigmoid()

#     # select masks corresponding to the predicted classes
#     num_masks = x.shape[0]
#     boxes_per_frame = [label.shape[0] for label in labels]
#     labels = torch.cat(labels)
#     index = torch.arange(num_masks, device=labels.device)
#     mask_prob = mask_prob[index, labels][:, None]
#     mask_prob = mask_prob.split(boxes_per_frame, dim=0)

#     return mask_prob


# def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
#     # type: (Tensor, Tensor, Tensor, int) -> Tensor
#     """
#     Given segmentation masks and the bounding boxes corresponding
#     to the location of the masks in the frame, this function
#     crops and resizes the masks in the position defined by the
#     boxes. This prepares the masks for them to be fed to the
#     loss computation as the targets.
#     """
#     matched_idxs = matched_idxs.to(boxes)
#     rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
#     gt_masks = gt_masks[:, None].to(rois)
#     return roi_align(gt_masks, rois, (M, M), 1.0)[:, 0]




# def _onnx_expand_boxes(boxes, scale):
#     # type: (Tensor, float) -> Tensor
#     w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
#     h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
#     x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
#     y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

#     w_half = w_half.to(dtype=torch.float32) * scale
#     h_half = h_half.to(dtype=torch.float32) * scale

#     boxes_exp0 = x_c - w_half
#     boxes_exp1 = y_c - h_half
#     boxes_exp2 = x_c + w_half
#     boxes_exp3 = y_c + h_half
#     boxes_exp = torch.stack((boxes_exp0, boxes_exp1, boxes_exp2, boxes_exp3), 1)
#     return boxes_exp


# # the next two functions should be merged inside Masker
# # but are kept here for the moment while we need them
# # temporarily for paste_mask_in_frame
# def expand_boxes(boxes, scale):
#     # type: (Tensor, float) -> Tensor
#     if torchvision._is_tracing():
#         return _onnx_expand_boxes(boxes, scale)
#     w_half = (boxes[:, 2] - boxes[:, 0]) * 0.5
#     h_half = (boxes[:, 3] - boxes[:, 1]) * 0.5
#     x_c = (boxes[:, 2] + boxes[:, 0]) * 0.5
#     y_c = (boxes[:, 3] + boxes[:, 1]) * 0.5

#     w_half *= scale
#     h_half *= scale

#     boxes_exp = torch.zeros_like(boxes)
#     boxes_exp[:, 0] = x_c - w_half
#     boxes_exp[:, 2] = x_c + w_half
#     boxes_exp[:, 1] = y_c - h_half
#     boxes_exp[:, 3] = y_c + h_half
#     return boxes_exp
    

class RoIHeads(nn.Module):
    __annotations__ = {
        "box_coder": BoxCoder,
        "proposal_matcher": Matcher,
        "fg_bg_sampler": BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        # box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_frame,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_frame,
    ):
        super().__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)

        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size_per_frame, positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = BoxCoder(bbox_reg_weights)

        # self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_frame = detections_per_frame

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        for proposals_in_frame, gt_boxes_in_frame, gt_labels_in_frame in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_frame.numel() == 0:
                # Background frame
                device = proposals_in_frame.device
                clamped_matched_idxs_in_frame = torch.zeros(
                    (proposals_in_frame.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_frame = torch.zeros((proposals_in_frame.shape[0],), dtype=torch.int64, device=device)
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_frame, proposals_in_frame)
                matched_idxs_in_frame = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_frame = matched_idxs_in_frame.clamp(min=0)

                labels_in_frame = gt_labels_in_frame[clamped_matched_idxs_in_frame]
                labels_in_frame = labels_in_frame.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_frame == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_frame[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_frame == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_frame[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_frame)
            labels.append(labels_in_frame)
        print("ASSIGN TARGETS TO PROPOSALS: ", len(labels), labels[0].shape)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [torch.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

        return proposals

    def select_training_samples(
        self,
        proposals,  # type: List[Tensor]
        targets,  # type: Optional[List[VoxelTarget]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t.boxes.to(dtype) for t in targets]
        gt_labels = [t.labels for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_frames = len(proposals)
        for img_id in range(num_frames):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_frame = gt_boxes[img_id]
            if gt_boxes_in_frame.numel() == 0:
                gt_boxes_in_frame = torch.zeros((1, 9), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_frame[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_frame = [boxes_in_frame.shape[0] for boxes_in_frame in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_frame, 0)
        pred_scores_list = pred_scores.split(boxes_per_frame, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores in zip(pred_boxes_list, pred_scores_list):

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 9)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        targets=None,  # type: Optional[List[VoxelTarget]]
    ):
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 9]])
            targets (List[VoxelTarget])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.float32, torch.double, torch.half)
                assert t.boxes.dtype in floating_point_types, "target boxes must of float type"
                assert t.labels.dtype == torch.int64, "target labels must of int64 type"

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
            print("LABELS FROM SELECTING TRAINING SAMPLES: ", len(labels), labels[0].shape)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        print("FEATURES: ", features["0"].shape)
        # box_features = self.box_roi_pool(features, proposals)
        # box_features = self.box_head(box_features)
        box_features = self.box_head(features["0"])
        print("BOX_FEATURES: ", box_features.shape)
        class_logits, box_regression = self.box_predictor(box_features)
        print("CLASS_LOGITS: ", class_logits.shape, "BOX_REGRESSION: ", box_regression.shape)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            # print("Labels: ", labels)
            # print("Regression Targets: ", regression_targets)
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals)
            num_frames = len(boxes)
            for i in range(num_frames):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, losses