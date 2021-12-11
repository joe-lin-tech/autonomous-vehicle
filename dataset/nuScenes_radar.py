import os
import numpy as np
import torch
from data_types.target import Target
from nuscenes.nuscenes import NuScenes


class nuScenesDataset(torch.utils.data.Dataset):
    def __init__(self, transforms):
        self.transforms = transforms
        self.nusc = NuScenes(version='v1.0-mini', dataroot='dataset/nuScenesDataset', verbose=True)

    def __getitem__(self, idx):
        # load pointclouds and object masks
        sample = self.nusc.sample[idx]
        print("Sample: ", sample)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open("object_mask_path")
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        target = Target(boxes=target["boxes"], labels=target["labels"], masks=target["masks"],
                        image_id=target["image_id"], area=target["area"], iscrowd=target["iscrowd"])

        return img, target

    def __len__(self):
        return len(self.nusc.scene)