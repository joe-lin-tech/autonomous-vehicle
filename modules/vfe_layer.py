from typing import List, Optional, Dict, Tuple, cast

import torch
import torchvision
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops

import utils._utils as det_utils


class VFELayer(torch.nn.Module):

    def __init__(self, cin, cout, T):
        super(VFELayer, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.T = T
        self.fcn = FCN(cin,self.units)

    def forward(self, x, mask):
        # point-wise feauture
        pwf = self.fcn(x)
        #locally aggregated feature
        laf = torch.max(pwf,1)[0].unsqueeze(1).repeat(1, self.T, 1)
        # point-wise concat feature
        pwcf = torch.cat((pwf,laf),dim=2)
        # apply mask
        mask = mask.unsqueeze(2).repeat(1, 1, self.units * 2)
        pwcf = pwcf * mask.float()

        return pwcf

    def forward(
        self,
        images: ImageList,
        features: Dict[str, Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:

        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[str, Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.
        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[str, Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        # print("Training: ", self.training)
        # TODO remove when working
        # self.training = True
        if self.training:
            # print("Targets: ", targets)
            assert targets is not None
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )
            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
            }
        return boxes, losses