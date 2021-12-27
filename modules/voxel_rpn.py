from typing import List, Optional, Dict, Tuple, cast

import torch
from torch.nn.modules import conv
import torchvision
from torch import nn, Tensor
from torch.nn import functional as F
from utils.voxel.ops import boxes as box_ops

import utils.voxel._utils as det_utils

# Import AnchorGenerator to keep compatibility.
from utils.voxel.anchor_utils import AnchorGenerator  # noqa: 401
from data_types.target import VoxelTarget


@torch.jit.unused
def _onnx_get_num_anchors_and_pre_nms_top_n(ob: Tensor, orig_pre_nms_top_n: int) -> Tuple[int, int]:
    from torch.onnx import operators

    num_anchors = operators.shape_as_tensor(ob)[1].unsqueeze(0)
    pre_nms_top_n = torch.min(torch.cat(
        (torch.tensor([orig_pre_nms_top_n], dtype=num_anchors.dtype), num_anchors), 0))

    # for mypy we cast at runtime
    return cast(int, num_anchors), cast(int, pre_nms_top_n)


class RPNBlock(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    Args:
        block_number: block position in RPN
    """

    def __init__(self, block_number: int) -> None:
        super().__init__()
        if block_number == 1:
            num_layers = 3
        else:
            num_layers = 5

        if block_number == 3:
            conv_layers = [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                           nn.BatchNorm2d(256),
                           nn.ReLU()]
        else:
            conv_layers = [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                           nn.BatchNorm2d(128),
                           nn.ReLU()]

        for _ in range(num_layers):
            if block_number == 3:
                conv_layers.append(
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
                conv_layers.append(nn.BatchNorm2d(256))
            else:
                conv_layers.append(
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
                conv_layers.append(nn.BatchNorm2d(128))
            conv_layers.append(nn.ReLU())

        self.conv_layers = nn.Sequential(*conv_layers)
        for layer in self.children():
            # type: ignore[arg-type]
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, nn.Conv2d):
                        torch.nn.init.normal_(sublayer.weight, std=0.01)
                        torch.nn.init.constant_(sublayer.bias, 0)
            else:
                torch.nn.init.normal_(layer.weight, std=0.01)
                # type: ignore[arg-type]
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_layers(x)


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads
    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels: int, num_anchors: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels,
                              kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors, kernel_size=1, stride=1)
        # TODO change output of bbox_pred
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 9, kernel_size=1, stride=1)

        for layer in self.children():
            # type: ignore[arg-type]
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

    # def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
    #     logits = []
    #     bbox_reg = []
    #     for feature in x:
    #         t = F.relu(self.conv(feature))
    #         logits.append(self.cls_logits(t))
    #         bbox_reg.append(self.bbox_pred(t))
    #     return logits, bbox_reg

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        t = F.relu(self.conv(x))
        logits = self.cls_logits(t)
        bbox_reg = self.bbox_pred(t)
        return logits, bbox_reg


def permute_and_flatten(layer: Tensor, N: int, A: int, C: int, H: int, W: int) -> Tensor:
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


# def concat_box_prediction_layers(box_cls: List[Tensor], box_regression: List[Tensor]) -> Tuple[Tensor, Tensor]:
#     box_cls_flattened = []
#     box_regression_flattened = []
#     # for each feature level, permute the outputs to make them be in the
#     # same format as the labels. Note that the labels are computed for
#     # all feature levels concatenated, so we keep the same representation
#     # for the objectness and the box_regression
#     print("BOX_CLS: ", len(box_cls), box_cls[0].shape)
#     print("BOX_REGRESSION: ", len(box_regression), box_regression[0].shape)
#     for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
#         N, AxC, H, W = box_cls_per_level.shape
#         Ax9 = box_regression_per_level.shape[1]
#         A = Ax9 // 9
#         C = AxC // A
#         box_cls_per_level = permute_and_flatten(
#             box_cls_per_level, N, A, C, H, W)
#         box_cls_flattened.append(box_cls_per_level)

#         box_regression_per_level = permute_and_flatten(
#             box_regression_per_level, N, A, 9, H, W)
#         box_regression_flattened.append(box_regression_per_level)
#     # concatenate on the first dimension (representing the feature levels), to
#     # take into account the way the labels were generated (with all feature maps
#     # being concatenated as well)
#     box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
#     box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 9)
#     return box_cls, box_regression


class RegionProposalNetwork(torch.nn.Module):
    """
    Implements Region Proposal Network (RPN).
    Args:
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): module that computes the objectness and regression deltas
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        batch_size_per_frame (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        pre_nms_top_n (Dict[str, int]): number of proposals to keep before applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        post_nms_top_n (Dict[str, int]): number of proposals to keep after applying NMS. It should
            contain two fields: training and testing, to allow for different values depending
            on training or evaluation
        nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
    """

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        head: nn.Module,
        # Faster-RCNN Training
        fg_iou_thresh: float,
        bg_iou_thresh: float,
        batch_size_per_frame: int,
        positive_fraction: float,
        # Faster-RCNN Inference
        pre_nms_top_n: Dict[str, int],
        post_nms_top_n: Dict[str, int],
        nms_thresh: float,
        score_thresh: float = 0.0,
    ) -> None:
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = det_utils.BoxCoder(
            weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))

        # used during training
        self.box_similarity = box_ops.box_iou

        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=True,
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_frame, positive_fraction)
        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3

        self.block_1 = RPNBlock(1)
        self.block_2 = RPNBlock(2)
        self.block_3 = RPNBlock(3)

        # self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(
        #     256, 256, 4, 4, 0), nn.BatchNorm2d(256))
        self.deconv_1 = nn.Sequential(nn.ConvTranspose2d(
            256, 256, 4, 4, 1), nn.BatchNorm2d(256))
        self.deconv_2 = nn.Sequential(nn.ConvTranspose2d(
            128, 256, 2, 2, 0), nn.BatchNorm2d(256))
        self.deconv_3 = nn.Sequential(nn.ConvTranspose2d(
            128, 256, 1, 1, 0), nn.BatchNorm2d(256))

    def pre_nms_top_n(self) -> int:
        if self.training:
            return self._pre_nms_top_n["training"]
        return self._pre_nms_top_n["testing"]

    def post_nms_top_n(self) -> int:
        if self.training:
            return self._post_nms_top_n["training"]
        return self._post_nms_top_n["testing"]

    def assign_targets_to_anchors(
        self, anchors: Tensor, targets: List[VoxelTarget]
    ) -> Tuple[List[Tensor], List[Tensor]]:

        labels = []
        matched_gt_boxes = []
        for anchors_per_frame, targets_per_frame in zip(anchors, targets):
            gt_boxes = targets_per_frame.boxes

            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_frame.device
                matched_gt_boxes_per_frame = torch.zeros(
                    anchors_per_frame.shape, dtype=torch.float32, device=device)
                labels_per_frame = torch.zeros(
                    (anchors_per_frame.shape[0],), dtype=torch.float32, device=device)
            else:
                print("gt_boxes: ", gt_boxes.shape)
                print("ANCHORS PER FRAME: ", anchors_per_frame.shape)
                anchors_per_frame = anchors_per_frame.flatten(1).transpose(0, 1)
                print("ANCHORS PER FRAME AFTER RESHAPE: ", anchors_per_frame.shape)
                match_quality_matrix = self.box_similarity(
                    gt_boxes, anchors_per_frame)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_frame = gt_boxes[matched_idxs.clamp(
                    min=0)]

                labels_per_frame = matched_idxs >= 0
                labels_per_frame = labels_per_frame.to(dtype=torch.float32)

                # Background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_frame[bg_indices] = 0.0

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_frame[inds_to_discard] = -1.0

            labels.append(labels_per_frame)
            matched_gt_boxes.append(matched_gt_boxes_per_frame)
        print("OUTPUT OF TARGETS TO ANCHORS: ", len(labels), len(matched_gt_boxes), labels[0].shape, matched_gt_boxes[0].shape)
        return labels, matched_gt_boxes

    def _get_top_n_idx(
        self,
        objectness: Tensor,
        # num_anchors_per_level: List[int]
    ) -> Tensor:
        # r = []
        # offset = 0
        # for ob in objectness.split(num_anchors_per_level, 1):
        #     if torchvision._is_tracing():
        #         num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(
        #             ob, self.pre_nms_top_n())
        #     else:
        #         num_anchors = ob.shape[1]
        #         pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
        #     _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
        #     r.append(top_n_idx + offset)
        #     offset += num_anchors
        # return torch.cat(r, dim=1)
        if torchvision._is_tracing():
            num_anchors, pre_nms_top_n = _onnx_get_num_anchors_and_pre_nms_top_n(
                objectness, self.pre_nms_top_n())
        else:
            num_anchors = objectness.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), num_anchors)
        _, top_n_idx = objectness.topk(pre_nms_top_n, dim=1)
        return top_n_idx

    def filter_proposals(
        self,
        proposals: Tensor,
        objectness: Tensor,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        num_frames = proposals.shape[0]
        device = proposals.device
        # do not backprop through objectness
        objectness = objectness.detach()
        # objectness = objectness.reshape(num_frames, -1)
        print("FILTER PROPOSALS: ", proposals.shape, objectness.shape)
        proposals = torch.flatten(torch.flatten(proposals, 2).transpose(1, 2), 2)
        objectness = torch.flatten(torch.flatten(objectness, 2).transpose(1, 2), 2)
        print("FILTER PROPOSALS AFTER FLATTEN: ", proposals.shape, objectness.shape)

        # levels = [
        #     torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
        # ]
        # levels = torch.cat(levels, 0)
        # levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        # top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)
        # top_n_idx = self._get_top_n_idx(objectness)

        # frame_range = torch.arange(num_frames, device=device)
        # batch_idx = frame_range[:, None]

        # objectness = objectness[batch_idx, top_n_idx]
        # # levels = levels[batch_idx, top_n_idx]
        # proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        # for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
        for boxes, scores in zip(proposals, objectness_prob):
            # boxes = box_ops.clip_boxes_to_image(boxes, img_shape)
            print("FILTERING PROPOSALS: ", boxes.shape, scores.shape)
            # remove small boxes
            keep = box_ops.remove_small_boxes(boxes, self.min_size)
            # boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            boxes, scores = boxes[keep], scores[keep]

            # TODO change later
            self.score_thresh = 0.57

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            print("KEEP HIGH SCORES: ", keep.shape)
            # boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]
            boxes, scores = boxes[keep], scores[keep]

            # non-maximum suppression, independently done per level
            # keep = box_ops.batched_nms(boxes, scores, lvl, self.nms_thresh)
            print("BEFORE NMS: ", boxes.shape, scores.shape)
            preserved_boxes, preserved_scores = box_ops.batched_nms(boxes, scores, self.nms_thresh)

            # keep only topk scoring predictions
            # keep = keep[: self.post_nms_top_n()]
            # boxes, scores = boxes[keep], scores[keep]
            boxes, scores = preserved_boxes[:self.post_nms_top_n()], preserved_scores[:self.post_nms_top_n()]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    def compute_loss(
        self, objectness: Tensor, pred_bbox_deltas: Tensor, labels: List[Tensor], regression_targets: List[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])
        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        print("PRED_BBOX_DELTAS: ", pred_bbox_deltas.shape)

        # TODO check implementation
        pred_bbox_deltas = torch.flatten(torch.flatten(torch.flatten(pred_bbox_deltas, 2).transpose(1, 2), 2), 0, 1)

        print("PRED_BBOX_DELTAS AFTER RESHAPE: ", pred_bbox_deltas.shape)
        print("LABELS: ", labels.shape)
        print("REG_TARGETS: ", regression_targets.shape)
        print("Sampled_inds: ", sampled_inds.shape)
        print("Sampled_pos_inds: ", sampled_pos_inds.shape)
        print("PRED_BBOX_DELTAS SAMPLED: ", pred_bbox_deltas[sampled_pos_inds].shape)
        print("LABELS SAMPLED: ", labels[sampled_inds].shape)
        print("REG_TARGETS SAMPLED: ", regression_targets[sampled_inds].shape)
        print("OBJECTNESS SAMPLED: ", objectness[sampled_inds].shape)

        box_loss = (
            F.smooth_l1_loss(
                pred_bbox_deltas[sampled_pos_inds],
                regression_targets[sampled_pos_inds],
                beta=1 / 9,
                reduction="sum",
            )
            / (sampled_inds.numel())
        )

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds])

        return objectness_loss, box_loss

    def forward(
        self,
        features: Tensor,
        targets: Optional[List[VoxelTarget]] = None,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """
        Args:
            features (Tensor): features computed from the frames that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.
        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                frame.
            losses (Dict[str, Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        features = self.block_1(features)
        print("BLOCK 1: ", features.shape)
        features_level_1 = features
        features = self.block_2(features)
        print("BLOCK 2: ", features.shape)
        features_level_2 = features
        features = self.block_3(features)
        print("BLOCK 3: ", features.shape)
        deconv_1 = self.deconv_1(features)
        print("DECONV 1: ", deconv_1.shape)
        deconv_2 = self.deconv_2(features_level_2)
        print("DECONV 2: ", deconv_2.shape)
        deconv_3 = self.deconv_3(features_level_1)
        print("DECONV 3: ", deconv_3.shape)
        features = torch.cat((deconv_1, deconv_2, deconv_3), 1)
        print("DECONV: ", features.shape)

        # objectness [batch_size, num_anchors_per_location, D, W]
        # pred_bbox_deltas [batch_size, num_anchors_per_location * 9, D, W]
        objectness, pred_bbox_deltas = self.head(features)
        print("OBJECTNESS: ", objectness.shape)
        print("PRED_BBOX_DELTAS: ", pred_bbox_deltas.shape)
        print("FEATURES: ", features.shape)

        # TODO unnecessary to pass in features
        anchors = self.anchor_generator(features)

        num_frames = anchors.shape[0]
        print("NUM_FRAMES: ", num_frames)

        # TODO remove unnecessary step
        # num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        # print("NUM_ANCHORS_PER_LEVEL_SHAPE_TENSORS: ", num_anchors_per_level_shape_tensors)
        # num_anchors_per_level = [s[0] * s[1] * s[2]
        #                          for s in num_anchors_per_level_shape_tensors]

        # TODO remove unnecessary concat (no levels)
        # objectness, pred_bbox_deltas = concat_box_prediction_layers(
        #     objectness, pred_bbox_deltas)

        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        print("PRED_BBOX_DELTAS: ", pred_bbox_deltas.shape)
        print("ANCHORS: ", anchors.shape)
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        # proposals = proposals.view(num_frames, -1, 9)
        print("PROPOSALS: ", proposals.shape)

        # TODO implement filtering proposals to output non-max suppressed bounding boxes
        boxes, scores = self.filter_proposals(
            proposals, objectness)

        print("PREDICTED BOXES: ", boxes[0][0])
        print("TARGETS: ", targets[0])

        losses = {}
        if self.training:
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(
                anchors, targets)
            print("ANCHORS: ", anchors.shape)
            regression_targets = self.box_coder.encode(
                matched_gt_boxes, anchors)
            print("REGRESSION_TARGETS AFTER ENCODE: ", regression_targets[0].shape, regression_targets[1].shape)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        # return boxes, losses
        return boxes, losses
