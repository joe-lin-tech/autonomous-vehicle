from collections import OrderedDict

from torch import nn
from torch._C import _from_dlpack
from torchvision.ops import MultiScaleRoIAlign
from utils.anchor_utils import AnchorGenerator
from modules.rpn import RPNHead
from modules.rpn import RegionProposalNetwork
import torch.nn.functional as F
from utils.roi_heads import RoIHeads
from utils.transform import GeneralizedRCNNTransform

class VoxelRCNN():
    def __init__(self, backbone, num_classes=None,
                 # transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 # Mask parameters
                 voxel_roi_pool=None, voxel_head=None, voxel_predictor=None):

        assert isinstance(voxel_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if voxel_predictor is not None:
                raise ValueError("num_classes should be None when mask_predictor is specified")

        out_channels = backbone.out_channels

        if voxel_roi_pool is None:
            voxel_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=14,
                sampling_ratio=2)

        if voxel_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            voxel_head = VoxelRCNNHeads(out_channels, mask_layers, mask_dilation)

        if voxel_predictor is None:
            voxel_predictor_in_channels = 256  # == mask_layers[-1]
            voxel_dim_reduced = 256
            voxel_predictor = VoxelRCNNPredictor(voxel_predictor_in_channels,
                                               voxel_dim_reduced, num_classes)

        # super(VoxelRCNN, self).__init__(
        #     backbone, num_classes,
        #     # transform parameters
        #     min_size, max_size,
        #     image_mean, image_std,
        #     # RPN-specific parameters
        #     rpn_anchor_generator, rpn_head,
        #     rpn_pre_nms_top_n_train, rpn_pre_nms_top_n_test,
        #     rpn_post_nms_top_n_train, rpn_post_nms_top_n_test,
        #     rpn_nms_thresh,
        #     rpn_fg_iou_thresh, rpn_bg_iou_thresh,
        #     rpn_batch_size_per_image, rpn_positive_fraction,
        #     rpn_score_thresh,
        #     # Box parameters
        #     box_roi_pool, box_head, box_predictor,
        #     box_score_thresh, box_nms_thresh, box_detections_per_img,
        #     box_fg_iou_thresh, box_bg_iou_thresh,
        #     box_batch_size_per_image, box_positive_fraction,
        #     bbox_reg_weights)

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")

        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2,
                representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

        self.roi_heads.voxel_roi_pool = voxel_roi_pool
        self.roi_heads.voxel_head = voxel_head
        self.roi_heads.voxel_predictor = voxel_predictor

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
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            # elif "bias" in name:
            #     nn.init.constant_(param, 0)

class VoxelRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, dim_reduced, num_classes):
        super(VoxelRCNNPredictor, self).__init__(OrderedDict([
            ("conv5_mask", nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)),
            ("relu", nn.ReLU(inplace=True)),
            ("mask_fcn_logits", nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)),
        ]))

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

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
        x = x.flatten(start_dim=1)

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
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas
