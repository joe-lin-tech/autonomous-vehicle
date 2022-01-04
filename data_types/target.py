from typing import NamedTuple
from torch import Tensor


class Target(NamedTuple):
    boxes: Tensor
    labels: Tensor
    masks: Tensor
    image_id: Tensor
    area: Tensor
    iscrowd: Tensor


class VoxelTarget(NamedTuple):
    boxes: Tensor
    labels: Tensor
    frame_id: Tensor
    volume: Tensor

    def tensor_to_list(self):
        return dict(boxes=self.boxes.tolist(), labels=self.labels.tolist(),
                    frame_id=self.frame_id.tolist(), volume=self.volume.tolist())
