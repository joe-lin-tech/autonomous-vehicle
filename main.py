import torch
from torch import nn, Tensor

import torchvision
from torch.utils.tensorboard import SummaryWriter
from models.voxel_rcnn import VoxelRCNN
from utils.anchor_utils import AnchorGenerator


class Backbone(nn.Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(self, scenes: Tensor, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
                else:
                    raise ValueError(
                        f"Expected target boxes to be of type Tensor, got {type(boxes)}.")
        return scenes.matmul(Tensor(1000, self.out_channels))


writer = SummaryWriter()

backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280

anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

# backbone = Backbone(1280)

roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'], output_size=7, sampling_ratio=2)
voxel_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
    featmap_names=['0'], output_size=14, sampling_ratio=2)
model = VoxelRCNN(backbone=backbone, num_classes=2, rpn_anchor_generator=anchor_generator,
                  box_roi_pool=roi_pooler, voxel_roi_pool=voxel_roi_pooler)
# model.eval()
input = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
output = model(input)
print(output)

writer.add_graph(model, input)
writer.close()
