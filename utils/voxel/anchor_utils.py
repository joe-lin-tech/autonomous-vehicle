import math
from typing import List, Optional

import torch
from torch import nn, Tensor, tensor
import numpy as np

from configs.config import BATCH_SIZE, RANGE_X, RANGE_Y, RANGE_Z, VOXEL_D, VOXEL_W, VOXEL_H, D, W, H


class AnchorGenerator(nn.Module):
    """
    Module that generates anchors for a set of feature maps and
    frame sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.

    sizes and aspect_ratios should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i] and aspect_ratios[i] can have an arbitrary number of elements,
    and AnchorGenerator will output a set of sizes[i] * aspect_ratios[i] anchors
    per spatial location for feature map i.

    Args:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
    """

    __annotations__ = {
        "cell_anchors": List[torch.Tensor],
    }

    def __init__(self):
        super().__init__()
        self.num_anchors_per_location = 1
        self.cell_anchors = self.generate_anchors()

    def generate_anchors(
        self,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ):

        # TODO add anchors at other sizes
        # x = np.linspace(xrange[0]+vw, xrange[1]-vw, W/2)
        # y = np.linspace(yrange[0]+vh, yrange[1]-vh, H/2)
        # cx, cy = np.meshgrid(x, y)
        # # all is (w, l, 2)
        # cx = np.tile(cx[..., np.newaxis], 2)
        # cy = np.tile(cy[..., np.newaxis], 2)
        # cz = np.ones_like(cx) * -1.0
        # w = np.ones_like(cx) * 1.6
        # l = np.ones_like(cx) * 3.9
        # h = np.ones_like(cx) * 1.56
        # r = np.ones_like(cx)
        # r[..., 0] = 0
        # r[..., 1] = np.pi/2
        # anchors = np.stack([cx, cy, cz, h, w, l, r], axis=-1)

        print("GENERATING ANCHORS X: ", RANGE_X[0] + VOXEL_D, RANGE_X[1] - VOXEL_D, D // 2)
        print("GENERATING ANCHORS Y: ", RANGE_Y[0] + VOXEL_W, RANGE_Y[1] - VOXEL_W, W // 2)

        x = np.linspace(RANGE_X[0] + VOXEL_D, RANGE_X[1] - VOXEL_D, D // 2)
        y = np.linspace(RANGE_Y[0] + VOXEL_W, RANGE_Y[1] - VOXEL_W, W // 2)

        xs, ys = np.meshgrid(x, y)
        # TODO (remove) use if generating two (self.num_anchors_per_location) anchors per feature map position
        # print("XS: ", xs.shape)
        # print("YS: ", ys.shape)
        # xs = np.tile(xs[..., np.newaxis], 2)
        # ys = np.tile(ys[..., np.newaxis], 2)

        # generating for multiple batches
        xs = np.tile(xs[np.newaxis, ...], (BATCH_SIZE, 1, 1))
        ys = np.tile(ys[np.newaxis, ...], (BATCH_SIZE, 1, 1))
        print("XS: ", xs.shape)
        print("YS: ", ys.shape)

        zs = np.ones_like(xs) * -1.0
        ds = np.ones_like(xs) * 3.9
        ws = np.ones_like(xs) * 1.6
        hs = np.ones_like(xs) * 1.56
        xrots = np.ones_like(xs) * 0
        yrots = np.ones_like(xs) * 0
        zrots = np.ones_like(xs) * 0

        xs = torch.as_tensor(xs, dtype=dtype)
        ys = torch.as_tensor(ys, dtype=dtype)
        zs = torch.as_tensor(zs, dtype=dtype)
        ds = torch.as_tensor(ds, dtype=dtype)
        ws = torch.as_tensor(ws, dtype=dtype)
        hs = torch.as_tensor(hs, dtype=dtype)
        xrots = torch.as_tensor(xrots, dtype=dtype)
        yrots = torch.as_tensor(yrots, dtype=dtype)
        zrots = torch.as_tensor(zrots, dtype=dtype)
        print("ALL SHAPES: ", xs.shape, ys.shape, zs.shape, ds.shape, ws.shape, hs.shape, xrots.shape, yrots.shape, zrots.shape)
        base_anchors = torch.stack(
            [xs, ys, zs, ds, ws, hs, xrots, yrots, zrots], dim=1)
        print("BASE ANCHORS: ", base_anchors.shape)
        return base_anchors.round()

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        # self.cell_anchors = [cell_anchor.to(
        #     dtype=dtype, device=device) for cell_anchor in self.cell_anchors]
        self.cell_anchors = self.cell_anchors.to(dtype=dtype, device=device)

    def forward(self, feature_maps: List[Tensor]) -> List[Tensor]:
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        self.set_cell_anchors(dtype, device)
        # anchors_over_all_feature_maps = self.grid_anchors()
        # anchors: List[List[torch.Tensor]] = []
        # for _ in range(len(feature_maps)):
        #     anchors_in_frame = [
        #         anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
        #     anchors.append(anchors_in_frame)
        # anchors = [torch.cat(anchors_per_frame)
        #            for anchors_per_frame in anchors]
        print("GENERATED ANCHORS: ", self.cell_anchors.shape)
        # anchors = [self.cell_anchors[0] for _ in range(len(feature_maps))]
        return self.cell_anchors
