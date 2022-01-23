import torch
from torch import nn, Tensor, tensor
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops
from model.modules.attention_block import SelfAttentionBlock

from configs.config import BATCH_SIZE, BEV_FEATURES, T, D, W, H, BACKBONE_OUT_CHANNELS, DROPOUT_RATE
import numpy as np

# 3D Convolution + Batch Normalization + Rectified Linear Unit
class Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, batch_norm=True):
        super(Conv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=k, stride=s, padding=p)
        if batch_norm:
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)

        return F.relu(x, inplace=True)

# Fully Connected Network
class FullyConnectedNetwork(nn.Module):
    def __init__(self, cin, cout):
        super(FullyConnectedNetwork, self).__init__()
        self.cout = cout
        self.linear = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        # KK is the stacked k across batch
        b, kk, t, _ = x.shape
        x = self.linear(x.view(b, kk * t, -1))
        x = F.relu(self.bn(x.view(b, self.cout, -1)))
        return x.view(b, kk, t, -1)

# Voxel Feature Encoding Layer
class VFELayer(torch.nn.Module):
    def __init__(self, cin, cout):
        super(VFELayer, self).__init__()
        assert cout % 2 == 0
        self.units = cout // 2
        self.fcn = FullyConnectedNetwork(cin, self.units)

    def forward(self, x, mask):
        # point-wise feature
        pwf = self.fcn(x)
        # locally aggregated feature
        laf = torch.max(pwf, 2)[0].unsqueeze(2).repeat(1, 1, T, 1)
        # point-wise concat feature
        pwcf = torch.cat((pwf, laf), dim=3)
        # apply mask
        mask = mask.unsqueeze(3).repeat(1, 1, 1, self.units * 2)
        pwcf = pwcf * mask.float()

        return pwcf

# Stacked Voxel Feature Encoder
class StackedVFE(nn.Module):
    def __init__(self):
        super(StackedVFE, self).__init__()
        self.vfe_1 = VFELayer(7, 32)
        self.vfe_2 = VFELayer(32, 128)
        self.fcn = FullyConnectedNetwork(128, 128)
    def forward(self, x):
        mask = torch.ne(torch.max(x, 3)[0], 0)
        x = self.vfe_1(x, mask)
        x = self.vfe_2(x, mask)
        x = self.fcn(x)
        # element-wise max pooling
        x = torch.max(x, 2)[0]
        return x

# Convolutional Middle Layer
class ConvolutionalMiddleLayer(nn.Module):
    def __init__(self):
        super(ConvolutionalMiddleLayer, self).__init__()
        self.conv_layers = nn.Sequential(
            Conv3d(128, 64, 3, s=(3, 1, 1), p=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            Conv3d(64, 64, 3, s=(2, 1, 1), p=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            Conv3d(64, 64, 3, s=(2, 1, 1), p=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            Conv3d(64, 64, 3, s=(3, 1, 1), p=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        # TODO result of conv_layers needs to be (2, 64, 2, D, W)
        # purpose of layers to extract and distribute features from z direction

    def forward(self, x):
        return self.conv_layers(x)


class VoxelBackbone(torch.nn.Module):
    def __init__(self):
        super(VoxelBackbone, self).__init__()
        self.svfe = StackedVFE()

        # added attention phase initialization
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        self.layer_norm = nn.LayerNorm(BEV_FEATURES, eps=1e-6)

        # Self-attention layers
        self.attention_1 = SelfAttentionBlock(inplanes=BEV_FEATURES, planes=BEV_FEATURES)
        self.attention_2 = SelfAttentionBlock(inplanes=BEV_FEATURES, planes=BEV_FEATURES)

        self.cml = ConvolutionalMiddleLayer()

    # omit if not needed
    def voxel_indexing(self, sparse_features, coords):
        dim = sparse_features.shape[-1]

        # TODO change for variable batch_size (dim 1)
        # TODO check implementation (correct feature extraction in correct dimensions - d, w, h)
        # dense_feature = torch.zeros(2, dim, D, W, H)

        # d = coords[..., 0]
        # w = coords[..., 1]
        # h = coords[..., 2]

        # dense_feature[0, :, d[0], w[0], h[0]] = sparse_features[0, ...].transpose(0, 1)
        # dense_feature[1, :, d[1], w[1], h[1]] = sparse_features[1, ...].transpose(0, 1)
        dense_feature = torch.zeros(BATCH_SIZE, dim, H, D, W)

        d = coords[..., 0]
        w = coords[..., 1]
        h = coords[..., 2]

        for batch_idx in range(BATCH_SIZE):
            dense_feature[batch_idx, :, h[batch_idx], d[batch_idx], w[batch_idx]] = sparse_features[batch_idx, ...].transpose(0, 1)

        return dense_feature
    
    def add_context_to_voxels(self, voxel_wise_features, voxel_coords):
        sparse_features = []

        for batch_idx in range(BATCH_SIZE):
            voxel_pillars = voxel_wise_features[batch_idx].unsqueeze(0)
            print("VOXEL_PILLARS: ", voxel_pillars.shape)
            voxel_pillars = voxel_pillars.permute(0, 2, 1).contiguous()

            voxel_pillars = self.attention_1(voxel_pillars)
            voxel_pillars = self.attention_2(voxel_pillars)
            voxel_pillars = voxel_pillars.permute(0, 2, 1).contiguous().squeeze(0)

            sparse_features.append(voxel_pillars)

        sparse_features = torch.stack(sparse_features, dim=0)
        print("SPARSE_FEATURES: ", sparse_features.shape)
        print("VOXEL_COORDS: ", voxel_coords.shape)
        context_features = self.voxel_indexing(sparse_features, voxel_coords)
        return context_features

    def forward(self, pointclouds):
        voxel_features, voxel_coords = pointclouds
        # feature learning network
        voxel_wise_features = self.svfe(voxel_features)

        # add back if necessary
        # voxel_wise_features = self.voxel_indexing(voxel_wise_features, voxel_coords)

        # added attention phase
        print("PRE-DROPOUT: ", voxel_wise_features.shape)
        voxel_wise_features = self.dropout(self.layer_norm(voxel_wise_features))
        print("VOXEL_WISE_FEATURES: ", voxel_wise_features.shape)
        print("VOXEL_COORDS: ", voxel_coords.shape)
        context_features = self.add_context_to_voxels(voxel_wise_features, voxel_coords)
        print("CONTEXTS: ", context_features.shape)

        # convolutional middle network
        # cml_out = self.cml(voxel_wise_features)
        print("BEFORE CML: ", context_features.shape)
        cml_out = self.cml(context_features)
        print("CML OUT: ", cml_out.shape)

        return cml_out.view(2, -1, D, W)
