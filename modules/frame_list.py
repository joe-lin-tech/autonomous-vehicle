
from typing import List, Tuple

import torch
from torch import Tensor


class FrameList:
    """
    Structure that holds a list of frames (of possibly
    varying sizes) as a single tensor.
    This works by padding the frames to the same size,
    and storing in a field the original sizes of each frame
    Args:
        tensors (tensor): Tensor containing frames.
        frame_sizes (list[tuple[int, int]]): List of Tuples each containing size of frames.
    """

    def __init__(self, tensors: Tensor, frame_sizes: List[Tuple[int, int]]) -> None:
        self.tensors = tensors
        self.frame_sizes = frame_sizes

    def to(self, device: torch.device) -> "FrameList":
        cast_tensor = self.tensors.to(device)
        return FrameList(cast_tensor, self.frame_sizes)