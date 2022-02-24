import os
import numpy as np
import torch
from data_types.target import VoxelTarget
from data_types.labels import LabelTypes
import csv
import json
from configs.config import T, VOXEL_D, VOXEL_W, VOXEL_H, RANGE_X, RANGE_Y, RANGE_Z


def preprocess_points(radar_pc):
    # shuffling the points
    np.random.shuffle(radar_pc)

    radar_pc = np.array(radar_pc)
    radar_pc = np.hstack((radar_pc[:, :3], radar_pc[:, 6:7]))

    voxel_coords = ((radar_pc[:, :3] - np.array([RANGE_X[0], RANGE_Y[0], RANGE_Z[0]])) / (
                    VOXEL_D, VOXEL_W, VOXEL_H)).astype(np.int32)


    # convert to  (D, H, W)
    # voxel_coords = voxel_coords[:,[2,1,0]]
    voxel_coords, inv_ind, voxel_counts = np.unique(voxel_coords, axis=0,
                                                    return_inverse=True, return_counts=True)

    voxel_features = []
    
    for i in range(len(voxel_coords)):
        voxel = np.zeros((T, 7), dtype=np.float32)
        pts = radar_pc[inv_ind == i]
        if voxel_counts[i] > T:
            pts = pts[:T, :]
            voxel_counts[i] = T
        # augment the points
        voxel[:pts.shape[0], :] = np.concatenate(
            (pts, pts[:, :3] - np.mean(pts[:, :3], 0)), axis=1)
        voxel_features.append(voxel)
    # return torch.as_tensor(voxel_features, dtype=torch.float32), torch.as_tensor(voxel_coords, dtype=torch.int32)
    return voxel_features, voxel_coords


class ZadarLabsDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.scene = "zadar_zsignal_dataset_1car_1bicycle_1human/segment-1"
        self.frames = list(sorted(os.listdir(os.path.join(
            os.getcwd(), "dataset/ZadarLabsDataset", self.scene))))[:-1]

    def __getitem__(self, idx):
        pc_path = os.path.join(
            os.getcwd(), "dataset/ZadarLabsDataset", self.scene, self.frames[idx], "zsignal0_zvue.csv")
        pc_points = []
        with open(pc_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            header = next(reader)
            for row in reader:
                pc_points.append(list(map(float, row[:5])))

        gt_path = os.path.join(
            os.getcwd(), "dataset/ZadarLabsDataset", self.scene, self.frames[idx], "gtruth_labels.json")
        with open(gt_path, 'r') as f:
            gt_json = json.load(f)
            radar_cuboids = gt_json["radar_cuboids"]
            labels = [
                LabelTypes[label.upper()].value for label in gt_json["labels"]]

        # pc_points = preprocess(pc_points)
        pc_points = torch.as_tensor(pc_points, dtype=torch.float32)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(radar_cuboids, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # frame_id = torch.tensor([idx])
        frame_id = torch.tensor([int(self.frames[idx])])
        volume = boxes[:, 3] * boxes[:, 4] * boxes[:, 5]

        target = VoxelTarget(boxes=boxes, labels=labels,
                             frame_id=frame_id, volume=volume)

        return pc_points, target

    def __len__(self):
        return len(self.frames)
