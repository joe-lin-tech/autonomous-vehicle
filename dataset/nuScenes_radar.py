import os
import numpy as np
import torch
from data_types.target import Target
from nuscenes.nuscenes import NuScenes
from typing import List
from nuscenes.utils.data_classes import RadarPointCloud
from typing import Tuple

import numpy as np
from tensorbay import GAS
from tensorbay.dataset import FusionDataset

from configs.dataset_config import GAS_KEY


class nuScenesDataset(torch.utils.data.Dataset):
    # TODO integrate all sensors: ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]
    # TODO integrate all attributes: ['x', 'y', 'z', 'dyn_prop', 'id', 'rcs', 'vx', 'vy', 'vx_comp', 'vy_comp',
    #                           'is_quality_valid', 'ambig_state', 'x_rms', 'y_rms', 'invalid_state', 'pdh0', 'vx_rms', 'vy_rms']
    def __init__(self, transforms, sensors: List = ["RADAR_FRONT"],
                 pc_attrs: List[Tuple] = [("x", 0), ("y", 1), ("z", 2), ("vx_comp", 8), ("vy_comp", 9)]):
        self.transforms = transforms
        self.sensors = sensors
        self.pc_attrs = pc_attrs
        # load from local
        self.nusc = NuScenes(version='v1.0-mini',
                             dataroot='dataset/nuScenesDataset', verbose=True)
        # TODO load from api
        # Authorize a GAS client.
        # gas = GAS(GAS_KEY)
        # print(gas)

        # # Get the fusion dataset.
        # fusion_dataset = FusionDataset("nuScenes", gas)
        # print("nuScenes Dataset: ", fusion_dataset)

        # # List fusion dataset segments
        # fusion_segments = fusion_dataset.keys()

        # # Get a segment by name
        # fusion_segment = fusion_dataset["v1.0-mini_scene-0061"]
        # for frame in fusion_segment:
        #     for sensor_name, data in frame.items():
        #         fp = data.open()
        #         # Use the data as you like.
        #         if sensor_name in self.sensors:
        #             for label_box3d in data.label.box3d:
        #                 size = label_box3d.size
        #                 translation = label_box3d.translation
        #                 rotation = label_box3d.rotation
        #                 box3d_category = label_box3d.category
        #                 box3d_attributes = label_box3d.attributes

    def __getitem__(self, idx):
        # load pointclouds and object bounding boxes
        sample = self.nusc.sample[idx]
        radar_data = {sensor: self.nusc.get(
            'sample_data', sample['data'][sensor]) for sensor in self.sensors}
        # TODO integrate pointcloud consolidation from sensors (defaulting to RADAR_FRONT)
        RadarPointCloud.disable_filters()
        pc = RadarPointCloud.from_file(os.path.join(
            'dataset/nuScenesDataset', radar_data["RADAR_FRONT"]['filename']))
        print("Radar Pointcloud Points: ", pc.points, "Size: ", pc.points.shape)
        pc_points = np.transpose(pc.points)[np.ix_([True] * len(np.transpose(pc.points)), [idx for _, idx in self.pc_attrs])]
        print("pc_points: ", pc_points)

        # # convert the PIL Image into a numpy array
        # mask = np.array(mask)
        # # instances are encoded as different colors
        # obj_ids = np.unique(mask)
        # # first id is the background, so remove it
        # obj_ids = obj_ids[1:]

        # # split the color-encoded mask into a set
        # # of binary masks
        # masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        # num_objs = len(obj_ids)
        # boxes = []
        # for i in range(num_objs):
        #     pos = np.where(masks[i])
        #     xmin = np.min(pos[1])
        #     xmax = np.max(pos[1])
        #     ymin = np.min(pos[0])
        #     ymax = np.max(pos[0])
        #     boxes.append([xmin, ymin, xmax, ymax])

        # # convert everything into a torch.Tensor
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # # there is only one class
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.uint8)

        # image_id = torch.tensor([idx])
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # # suppose all instances are not crowd
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # target = {}
        # target["boxes"] = boxes
        # target["labels"] = labels
        # target["masks"] = masks
        # target["image_id"] = image_id
        # target["area"] = area
        # target["iscrowd"] = iscrowd

        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        # target = Target(boxes=target["boxes"], labels=target["labels"], masks=target["masks"],
        #                 image_id=target["image_id"], area=target["area"], iscrowd=target["iscrowd"])

        # return img, target
        return None

    def __len__(self):
        return len(self.nusc.scene)
