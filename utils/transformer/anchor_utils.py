import torch
from torch import nn
from configs.config import BATCH_SIZE, RANGE_X, RANGE_Y, RANGE_Z, VOXEL_D, VOXEL_W, VOXEL_H, D, W, H


class AnchorGenerator(nn.Module):
    def __init__(self):
        super(AnchorGenerator, self).__init__()

    def generate_anchors(self):
        x = torch.linspace(RANGE_X[0] + VOXEL_D, RANGE_X[1] - VOXEL_D, D // 2)
        y = torch.linspace(RANGE_Y[0] + VOXEL_W, RANGE_Y[1] - VOXEL_W, W // 2)
        xs, ys = torch.meshgrid(x, y)
        # TODO generate multiple anchors per feature map location
        xs = torch.tile(xs[None, ...], (BATCH_SIZE, 1, 1))
        ys = torch.tile(ys[None, ...], (BATCH_SIZE, 1, 1))
        zs = torch.ones_like(xs) * -1.0
        ds = torch.ones_like(xs) * 3.9
        ws = torch.ones_like(xs) * 1.6
        hs = torch.ones_like(xs) * 1.56
        xrots = torch.ones_like(xs) * 0
        yrots = torch.ones_like(xs) * 0
        zrots = torch.ones_like(xs) * 0
        base_anchors = torch.stack(
            [xs, ys, zs, ds, ws, hs, xrots, yrots, zrots], dim=1)
        return base_anchors.round()

    def forward(self, dtype, device):
        anchors = self.generate_anchors().to(dtype=dtype, device=device)
        return anchors
