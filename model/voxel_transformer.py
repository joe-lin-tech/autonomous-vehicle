from sympy import N
import torch
from torch import nn, Tensor
from torch.nn.functional import softmax, relu, sigmoid, binary_cross_entropy_with_logits, smooth_l1_loss, max_pool3d
from configs.transformer_config import BATCH_SIZE, D, W, H
from utils.transformer.anchor_utils import AnchorGenerator
from utils.transformer.box_ops import bev_iou, generalized_iou, batched_nms
import utils.transformer._utils as utils


class InputEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(InputEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        return self.linear(x)


class LinearBatchReLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearBatchReLU, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class SetFeatureAbstraction(nn.Module):
    def __init__(self, feature_dim, spatial_dim, k):
        super(SetFeatureAbstraction, self).__init__()
        self.feature_dim = feature_dim
        self.spatial_dim = spatial_dim
        self.k = k

    def farthest_point_sampling(self, spatial):
        n = spatial.shape[1]
        batched_sample_idxs = torch.empty(BATCH_SIZE)
        for b in range(BATCH_SIZE):
            points = spatial[b]
            # points: [n, 5]
            points_remaining = torch.arange(n)
            # points_remaining: [n, 1]
            sample_idxs = torch.zeros(n / 2, dtype=torch.int)
            # sample_idxs: [k, 1]
            distances = torch.ones_like(points_remaining) * float('inf')
            # distances: [n, 1]
            selected_idx = 0
            sample_idxs[0] = points_remaining[selected_idx]
            points_remaining = points_remaining[points_remaining != selected_idx]
            # points_remaining: [n-1, 1]
            for i in range(1, n / 2):
                added_idx = sample_idxs[i - 1]
                distance_to_last_added_point = torch.sum(
                    (points[added_idx] - points[points_remaining]) ** 2, dim=-1)
                # distance_to_last_added_point: [n-i, 1]
                distances[points_remaining] = torch.min(
                    distance_to_last_added_point, distances[points_remaining])
                # distances: [n-i, 1]
                selected_idx = torch.argmax(distances[points_remaining])
                sample_idxs[i] = points_remaining[selected_idx]
                points_remaining = points_remaining[points_remaining != selected_idx]
            batched_sample_idxs[b] = sample_idxs
        return batched_sample_idxs

    def k_nearest_neighbors(self, x, spatial, batched_sample_idxs):
        # x: [B, n/2, feature_dim], spatial: [B, n, 5], batched_sample_idxs: [B, n/2]
        neighbors_feature = torch.empty(BATCH_SIZE, self.k, self.feature_dim)
        neighbors_spatial = torch.empty(BATCH_SIZE, self.k, 5)
        for b in BATCH_SIZE:
            sample_idxs = batched_sample_idxs[b]
            for i in sample_idxs:
                distances = torch.zeros(j)
                for j in x[b].shape[0]:
                    distances[j] = torch.sqrt(
                        torch.sum((spatial[b][i] - spatial[b][j]) ** 2, dim=-1))
                _, k_nearest_idxs = torch.topk(distances, self.k + 1)
                if not i in k_nearest_idxs[:self.k + 1]:
                    k_nearest_idxs = k_nearest_idxs[:self.k]
                else:
                    k_nearest_idxs = k_nearest_idxs[k_nearest_idxs != i]
                neighbors_feature[b][:self.k] = x[b][k_nearest_idxs]
                neighbors_spatial[b][:self.k] = spatial[b][k_nearest_idxs]
        return neighbors_feature, neighbors_spatial

    def forward(self, x, spatial):
        # x: [B, n, feature_dim], spatial: [B, n, 5]
        batched_sample_idxs = self.farthest_point_sampling(x, spatial)
        # batched_sample_idxs: [B, n/2]
        x = x[batched_sample_idxs]
        # x: [B, n/2, feature_dim]
        spatial = spatial[batched_sample_idxs]
        # spatial: [B, n/2, 5]
        centroids = torch.tile(torch.unsqueeze(spatial, 1), (1, self.k, 1))
        neighbors_feature, neighbors_spatial = self.k_nearest_neighbors(
            x, spatial, batched_sample_idxs)
        # neighbors_feature: [B, n/2, k, feature_dim], neighbors_spatial: [B, n/2, k, 5]

        return x


class ScalarAttentionModule(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super(ScalarAttentionModule, self).__init__()
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.query_linear = nn.Linear(feature_dim, attention_dim)
        self.key_linear = nn.Linear(feature_dim, attention_dim)
        self.value_linear = nn.Linear(feature_dim, attention_dim)
        self.lbr = LinearBatchReLU(attention_dim, feature_dim)

    def forward(self, x):
        q = self.query_linear(x)
        k = self.key_linear(x)
        v = self.value_linear(x)

        weights = softmax(torch.bmm(q, torch.transpose(k, 1, 2)))
        feature = torch.bmm(weights, v)
        out = torch.add(self.lbr(feature), x)
        return out


class VectorAttentionModule(nn.Module):
    def __init__(self, feature_dim, attention_dim):
        super(VectorAttentionModule, self).__init__()
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.in_linear = nn.Linear(feature_dim, attention_dim)
        self.query_linear = nn.Linear(attention_dim, attention_dim)
        self.key_linear = nn.Linear(attention_dim, attention_dim)
        self.value_linear = nn.Linear(attention_dim, attention_dim)
        self.weights_linear_1 = nn.Linear(attention_dim, attention_dim)
        self.weights_relu = nn.ReLU(inplace=True)
        self.weights_linear_2 = nn.Linear(attention_dim, attention_dim)
        self.out_linear = nn.Linear(attention_dim, feature_dim)

    def forward(self, x):
        transform = self.in_linear(x)
        q = self.query_linear(transform)
        k = self.key_linear(transform)
        v = self.value_linear(transform)

        weights = softmax(self.weights_linear_2(
            self.weights_relu(self.weights_linear_1(torch.subtract(q, k)))))
        # TODO check for hadamard product
        feature = weights * v
        out = torch.add(self.out_linear(feature), x)
        return out


class LocalHierarchicalFeature(nn.Module):
    def __init__(self):
        super(LocalHierarchicalFeature, self).__init__()
        # TODO fix set abstraction and vector attention dimensions
        self.set_abstraction = SetFeatureAbstraction(
            feature_dim=64, spatial_dim=5, k=3)
        self.vector_attention = VectorAttentionModule(
            feature_dim=64, attention_dim=64)

    def forward(self, x):
        x = self.set_abstraction(x)
        x = self.vector_attention(x)
        return x


class RegionProposalNetwork(nn.Module):
    def __init__(self, feature_dim, num_anchors, score_threshold, nms_threshold):
        super(RegionProposalNetwork, self).__init__()
        self.anchor_generator = AnchorGenerator()
        print("GENERATED ANCHORS")
        print("FEATURE DIMENSION: ", feature_dim)
        self.rpn_conv = nn.Conv2d(
            feature_dim, feature_dim, kernel_size=3, stride=1, padding=1)
        print("INTIALIZED RPN CONV")
        self.classification_head = nn.Conv2d(
            feature_dim, num_anchors, kernel_size=1, stride=1)
        self.regression_head = nn.Conv2d(
            feature_dim, num_anchors * 9, kernel_size=1, stride=1)
        print("INTIALIZED HEADS")
        self.box_coder = utils.BoxCoder(
            weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
        print("INITIALIZED BOX CODER")
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.proposal_matcher = utils.Matcher(0.6, 0.2, True)

    def filter_proposals(self, proposals, objectness):
        """
        Filters proposals by removing those with a low score (classification score)
        and performs NMS on remaining proposals.

        Args:
            proposals: (N, 9) tensor where N is the number of proposals.
            objectness: (N, 1) tensor where N is the number of proposals.

        Returns:
            filtered_boxes: (N, 9) tensor where N is the number of remaining proposals.
            filtered_scores: (N, 1) tensor where N is the number of remaining proposals.
        """
        objectness = sigmoid(objectness)

        filtered_boxes, filtered_scores = [], []
        for boxes, scores in zip(proposals, objectness):
            # TODO implement remove small boxes
            keep = torch.where(scores >= self.score_threshold)[0]
            boxes, scores = boxes[keep], scores[keep]

            nms_boxes, nms_scores = batched_nms(
                boxes, scores, self.nms_threshold)

            # TODO implement taking only topk scoring boxes
            filtered_boxes.append(nms_boxes)
            filtered_scores.append(nms_scores)

        return filtered_boxes, filtered_scores

    def assign_targets_to_anchors(self, anchors, targets):
        """
        Assigns ground truth boxes and class labels to each anchor.

        Args:
            anchors: (B, M, 9) tensor where N is the number of anchors.
            targets: (B, N, 9) tensor where N is the number of targets.

        Returns:
            regression_targets: (B, M, 9) tensor with anchor regression targets.
            labels: (B, M, 1) tensor with anchor labels.
        """
        matched_gt_boxes, labels = [], []
        for anchors_per_frame, targets_per_frame in zip(anchors, targets):
            # anchors_per_frame: [M, 9]
            groundtruth_boxes = targets_per_frame[:, :9]
            # groundtruth_boxes: [N, 9]
            match_quality_matrix = bev_iou(
                groundtruth_boxes, anchors_per_frame)
            # match_quality_matrix: [N, M]
            matched_idxs = self.proposal_matcher(match_quality_matrix)
            matched_gt_boxes_per_frame = groundtruth_boxes[matched_idxs.clamp(
                min=0)]
            matched_gt_boxes.append(matched_gt_boxes_per_frame)

            labels_per_frame = matched_idxs >= 0
            labels_per_frame = labels_per_frame.to(dtype=torch.float32)
            bg_idxs = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_frame[bg_idxs] = 0.0
            discard_idxs = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_frame[discard_idxs] = -1.0
            labels.append(labels_per_frame)
        return matched_gt_boxes, labels

    def compute_loss(self, objectness, labels, regression_deltas, regression_targets):
        """
        Computes the classification and regression loss.

        Args:
            objectness: (B, M, 1) tensor with objectness scores.
            labels: (B, M, 1) tensor with anchor labels.
            regression_deltas: (B, M, 9) tensor with regression deltas.
            regression_targets: (B, M, 9) tensor with regression targets.

        Returns:
            classification_loss: (B, 1) tensor with classification loss.
            regression_loss: (B, 1) tensor with regression loss.
        """
        classification_loss = binary_cross_entropy_with_logits(
            objectness, labels, reduction='sum')
        regression_loss = smooth_l1_loss(
            regression_deltas, regression_targets, reduction='sum')
        return classification_loss, regression_loss

    def forward(self, x, targets):
        anchors = self.anchor_generator(x)
        # x: [B, H·512, D, W]
        x = relu(self.rpn_conv(x))
        objectness = self.classification_head(x)
        regression_deltas = self.regression_head(x)

        proposals = self.box_coder.decode(regression_deltas.detach(), anchors)
        detections = self.filter_proposals(proposals, objectness.detach())

        losses = {}
        if self.training:
            matched_gt_boxes, labels = self.assign_targets_to_anchors(
                anchors, targets)
            regression_targets = self.box_coder.encode(
                matched_gt_boxes, anchors)
            loss_objectness, loss_regression = self.compute_loss(
                objectness, labels, regression_deltas, regression_targets)
            losses = {
                'loss_objectness': loss_objectness,
                'loss_regression': loss_regression
            }

        return detections, losses


class VoxelTransformer(nn.Module):
    def __init__(self):
        super(VoxelTransformer, self).__init__()
        # TODO insert feature parameters for local hierarchical features
        self.input_embedding = InputEmbedding(input_dim=5, embedding_dim=64)
        print("INITIALIZED INPUT EMBEDDING")
        self.vector_attention1 = VectorAttentionModule(
            feature_dim=64, attention_dim=64)
        self.local_hierarchical1 = LocalHierarchicalFeature()
        print("INITIALIZED SECTION 1")
        self.vector_attention2 = VectorAttentionModule(
            feature_dim=128, attention_dim=128)
        self.local_hierarchical2 = LocalHierarchicalFeature()
        print("INITIALIZED SECTION 2")
        self.vector_attention3 = VectorAttentionModule(
            feature_dim=256, attention_dim=256)
        self.local_hierarchical3 = LocalHierarchicalFeature()
        print("INITIALIZED SECTION 3")
        self.scalar_attention = ScalarAttentionModule(
            feature_dim=512, attention_dim=512)
        print("INTIALIZED SCALAR ATTENTION")
        self.aggregate_conv = nn.Conv2d(
            512, 64, kernel_size=1, stride=1)
        print("INITIALIZED AGGREGATE CONV")
        self.region_proposal_network = RegionProposalNetwork(
            feature_dim=64 * H, num_anchors=1, score_threshold=0.65, nms_threshold=0.5)
        print("INITIALIZED REGION PROPOSAL NETWORK")

    def generate_voxel_tensor(self, features, coordinates):
        """
        Generates a voxel tensor from a feature tensor and a coordinate tensor.

        Args:
            features: (B, N, C) where N is the number of features and C is the feature dimension.
            coordinates: (B, N, 3) where N is the number of features and the last dimension is the coordinates of the voxels.

        Returns:
            voxel_tensor: (B, H, D, W, C) where H is the height, D is the depth, W is the width, and C is the feature dimension.
        """
        feature_dim = features.shape[2]
        voxel_tensor = torch.zeros(BATCH_SIZE, H, D, W, feature_dim)

        d = coordinates[..., 0]
        w = coordinates[..., 1]
        h = coordinates[..., 2]

        for batch_idx in range(BATCH_SIZE):
            voxel_tensor[batch_idx, h[batch_idx], d[batch_idx],
                         w[batch_idx], :] = torch.transpose(features[batch_idx, ...], 0, 1)

        return voxel_tensor

    def forward(self, x, y):
        # x: [B, N, 5]
        features = self.input_embedding(x)
        print("COMPLETED INPUT EMBEDDING...")
        # features: [B, N, 64]
        spatial = x
        # spatial: [B, N, 5]
        coordinates = x[:3]
        # coordinates: [B, N, 3]
        x = self.vector_attention1(features)
        print("APPLIED VECTOR ATTENTION...")
        # x: [B, H, D, W, 64]
        local, spatial = self.local_hierarchical1(x, spatial)
        # local: [B, N/2, 64], spatial: [B, N/2, 5]
        # TODO check for concatenation and tiling
        # local should be tiled from [B, N/2, 64] to [B, N, 64]
        # concatenate should be [B, N, 64] + [B, N, 64] = [B, N, 128]
        x = torch.cat([x, torch.tile(local, (1, 2, 1))], dim=2)
        x = self.vector_attention2(x)
        # x: [B, N, 128]
        local, spatial = self.local_hierarchical2(local, spatial)
        # local: [B, N/4, 128], spatial: [B, N/4, 5]
        x = torch.cat([x, torch.tile(local, (1, 4, 1))], dim=2)
        # x: [B, N, 256]
        x = self.vector_attention3(x)
        # x: [B, N, 256]
        local, spatial = self.local_hierarchical3(local, spatial)
        # local: [B, N/8, 256], spatial: [B, N/8, 5]
        x = torch.cat([x, torch.tile(local, (1, 8, 1))], dim=2)
        # x: [B, N, 512]
        x = self.scalar_attention(x)
        # x: [B, N, 512]
        x = self.aggregate_conv(x)
        # x: [B, N, 64]
        x, spatial = self.generate_voxel_tensor(features, coordinates)
        x = x.view(BATCH_SIZE, -1, D, W)
        # x: [B, H·64, D, W]
        detections, losses = self.region_proposal_network(x)
        # detections: [B, N, 9], [B, N, 1]
        # losses: {'loss_objectness': loss_objectness, 'loss_regression': loss_regression}
        return detections, losses
