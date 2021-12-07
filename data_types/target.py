from typing import NamedTuple
from torch import Tensor

class Target(NamedTuple):
    boxes: Tensor
    labels: Tensor
    masks: Tensor