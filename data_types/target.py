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