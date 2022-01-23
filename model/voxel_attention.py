from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch._C import _from_dlpack
from torchvision.ops import MultiScaleRoIAlign
from utils.voxel.anchor_utils import AnchorGenerator
from model.modules.voxel_rpn import RPNHead, RegionProposalNetwork
import torch.nn.functional as F
import warnings

from typing import List, Tuple, Optional
from data_types.target import VoxelTarget
import math


class VoxelAttention(nn.Module):
    def __init__(self, backbone, num_classes=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 # originally fg: 0.7; bg: 0.3
                 rpn_fg_iou_thresh=0.5, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_frame=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))

        if rpn_anchor_generator is None:
            rpn_anchor_generator = AnchorGenerator()
        if rpn_head is None:
            rpn_head = RPNHead(
                768, rpn_anchor_generator.num_anchors_per_location
            )

        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_frame, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        super().__init__()
        self.backbone = backbone
        self.rpn = rpn

        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections, write_graph):
        print("Eager Outputs: ", losses, type(losses))
        if write_graph:
            return Tensor([list(losses.items())])

        if self.training:
            return losses, detections

        return detections

    def forward(self, pointclouds: List[Tensor], targets: Optional[List[VoxelTarget]] = None, write_graph: Tensor = torch.BoolTensor([False])):
        """
        Args:
            pointclouds (list[Tensor]): pointclouds to be processed
            targets (list[VoxelTarget]): ground-truth boxes present in the frame (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if not isinstance(targets[0], VoxelTarget):
            for i, t in enumerate(targets):
                targets[i] = VoxelTarget(
                    boxes=t[0], labels=t[1], frame_id=t[2], volume=t[3])
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target.boxes
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 9:
                        raise ValueError(
                            f"Expected target boxes to be a tensor of shape [N, 9], got {boxes.shape}.")
                else:
                    raise ValueError(
                        f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        voxel_encoding = self.backbone(pointclouds)

        detections, detection_losses = self.rpn(voxel_encoding, targets)

        # detection_boxes, detection_scores = detections

        losses = {}
        losses.update(detection_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn(
                    "VoxelAttention always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            # return losses, detections
            return losses
        else:
            # return self.eager_outputs(losses, detections, write_graph=write_graph)
            return self.eager_outputs(losses, detections, write_graph=write_graph)
