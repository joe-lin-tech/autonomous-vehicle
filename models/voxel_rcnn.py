from collections import OrderedDict

import torch
from torch import nn, Tensor
from torch._C import _from_dlpack
from torchvision.ops import MultiScaleRoIAlign
from utils.voxel.anchor_utils import AnchorGenerator
from modules.voxel_rpn import RPNHead, RegionProposalNetwork
import torch.nn.functional as F
from utils.voxel.roi_heads import RoIHeads
import warnings

from typing import List, Tuple, Optional
from data_types.target import VoxelTarget


class VoxelRCNN(nn.Module):
    def __init__(self, backbone, num_classes=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_frame=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 #  box_roi_pool=None,
                 box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_frame=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_frame=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        # assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError(
                    "num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

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

        if box_head is None:
            representation_size = 1024
            box_head = TwoMLPHead(
                10000,
                # out_channels,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        roi_heads = RoIHeads(
            # Box
            # box_roi_pool,
            box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_frame, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_frame)

        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections, write_graph):
        print("Eager Outputs: ", losses, type(losses))
        if write_graph:
            return Tensor([list(losses.items())])

        if self.training:
            return losses

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

        features = self.backbone(pointclouds)
        print("FEATURES AFTER BACKBONE: ", features.shape)
        # if isinstance(features, torch.Tensor):
        #     features = OrderedDict([("0", features)])
        print("TARGETS: ", len(targets))

        proposals, proposal_losses = self.rpn(features, targets)
        # detections, detector_losses = self.roi_heads(
        #     features, proposals, targets)

        losses = {}
        # losses.update(detector_losses)
        losses.update(proposal_losses)

        # print("Detector Losses: ", detector_losses)
        print("Proposal Losses: ", proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn(
                    "RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            # return losses, detections
            return losses
        else:
            # return self.eager_outputs(losses, detections, write_graph=write_graph)
            return self.eager_outputs(losses, None, write_graph=write_graph)


class VoxelRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        """
        Args:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
        """
        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d["mask_fcn{}".format(layer_idx)] = nn.Conv2d(
                next_feature, layer_features, kernel_size=3,
                stride=1, padding=dilation, dilation=dilation)
            d["relu{}".format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        super(VoxelRCNNHeads, self).__init__(d)
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(
                    param, mode="fan_out", nonlinearity="relu")


class VoxelRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super(VoxelRCNNPredictor, self).__init__(OrderedDict([
            ("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
            ("relu", nn.ReLU(inplace=True)),
            ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
        ]))

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(
                    param, mode="fan_out", nonlinearity="relu")


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        # x = x.flatten(start_dim=1)
        x = x.flatten(start_dim=2)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 9)

    def forward(self, x):
        print("X DIM: ", x.dim())
        # x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas
