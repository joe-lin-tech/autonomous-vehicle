
import math
from collections import OrderedDict
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torchvision.ops.misc import FrozenBatchNorm2d

import datetime
import errno
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist
import re
from collections import defaultdict


class BalancedPositiveNegativeSampler:
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_frame: int, positive_fraction: float) -> None:
        """
        Args:
            batch_size_per_frame (int): number of elements to be selected per frame
            positive_fraction (float): percentage of positive elements per batch
        """
        self.batch_size_per_frame = batch_size_per_frame
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Args:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific frame.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.
        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])
        Returns two lists of binary masks for each frame.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_frame in matched_idxs:
            positive = torch.where(matched_idxs_per_frame >= 1)[0]
            negative = torch.where(matched_idxs_per_frame == 0)[0]

            num_pos = int(self.batch_size_per_frame * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_frame - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            perm1 = torch.randperm(
                positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(
                negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_frame = positive[perm1]
            neg_idx_per_frame = negative[perm2]

            # create binary mask from indices
            pos_idx_per_frame_mask = torch.zeros_like(
                matched_idxs_per_frame, dtype=torch.uint8)
            neg_idx_per_frame_mask = torch.zeros_like(
                matched_idxs_per_frame, dtype=torch.uint8)

            pos_idx_per_frame_mask[pos_idx_per_frame] = 1
            neg_idx_per_frame_mask[neg_idx_per_frame] = 1

            pos_idx.append(pos_idx_per_frame_mask)
            neg_idx.append(neg_idx_per_frame_mask)

        return pos_idx, neg_idx


@torch.jit._script_if_tracing
def encode_boxes(reference_boxes: Tensor, proposals: Tensor, weights: Tensor) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes
    Args:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
        weights (Tensor[9]): the weights for ``(x, y, z, d, w, h, x_rot, y_rot, z_rot)``
    """
    print("ENCODE BOXES: ", reference_boxes.shape, proposals.shape, weights.shape)
    # perform some unpacking to make it JIT-fusion friendly
    wx, wy, wz, wd, ww, wh, wx_rot, wy_rot, wz_rot = weights

    proposals_x = proposals[:, 0].unsqueeze(1)
    proposals_y = proposals[:, 1].unsqueeze(1)
    proposals_z = proposals[:, 2].unsqueeze(1)
    proposals_d = proposals[:, 3].unsqueeze(1)
    proposals_w = proposals[:, 4].unsqueeze(1)
    proposals_h = proposals[:, 5].unsqueeze(1)
    proposals_x_rot = proposals[:, 6].unsqueeze(1)
    proposals_y_rot = proposals[:, 7].unsqueeze(1)
    proposals_z_rot = proposals[:, 8].unsqueeze(1)

    reference_boxes_x = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_z = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_d = reference_boxes[:, 3].unsqueeze(1)
    reference_boxes_w = reference_boxes[:, 4].unsqueeze(1)
    reference_boxes_h = reference_boxes[:, 5].unsqueeze(1)
    reference_boxes_x_rot = reference_boxes[:, 6].unsqueeze(1)
    reference_boxes_y_rot = reference_boxes[:, 7].unsqueeze(1)
    reference_boxes_z_rot = reference_boxes[:, 8].unsqueeze(1)

    # implementation starts here
    diagonal = torch.sqrt(proposals_d ** 2 + proposals_w ** 2)
    targets_dx = wx * (reference_boxes_x - proposals_x) / diagonal
    targets_dy = wy * (reference_boxes_y - proposals_y) / diagonal
    targets_dz = wz * (reference_boxes_z - proposals_z) / proposals_h
    targets_dd = wd * torch.log(reference_boxes_d / proposals_d)
    targets_dw = ww * torch.log(reference_boxes_w / proposals_w)
    targets_dh = wh * torch.log(reference_boxes_h / proposals_h)
    targets_dx_rot = wx_rot * (reference_boxes_x_rot - proposals_x_rot)
    targets_dy_rot = wy_rot * (reference_boxes_y_rot - proposals_y_rot)
    targets_dz_rot = wz_rot * (reference_boxes_z_rot - proposals_z_rot)

    targets = torch.cat(
        (targets_dx, targets_dy, targets_dz, targets_dd, targets_dw, targets_dh, targets_dx_rot, targets_dy_rot, targets_dz_rot), dim=1)
    return targets


class BoxCoder:
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(
        self,
        weights: Tuple[float, float, float, float, float, float, float, float, float],
        bbox_xform_clip: float = math.log(1000.0 / 16)
    ) -> None:
        """
        Args:
            weights (9-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes: List[Tensor], proposals: List[Tensor]) -> List[Tensor]:
        boxes_per_frame = [len(b) for b in reference_boxes]
        print("BOXES_PER_FRAME: ", boxes_per_frame)
        print("REFERENCE BOXES: ", reference_boxes[0].shape, reference_boxes[1].shape, type(reference_boxes))
        reference_boxes = torch.cat((reference_boxes), dim=0)
        print("REFERENCE BOXES: ", reference_boxes.shape)
        print("PROPOSALS: ", proposals.shape, type(proposals))
        # proposals = torch.cat((proposals), dim=0)
        proposals = torch.flatten(torch.flatten(torch.flatten(proposals, 2).transpose(1, 2), 2), 0, 1)
        print("PROPOSALS: ", proposals, proposals.shape)
        regression_targets = self.encode_single(reference_boxes, proposals)
        return regression_targets.split(boxes_per_frame, 0)

    def encode_single(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
        """
        Encode a set of proposals with respect to some
        reference boxes
        Args:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    # TODO originally boxes is List[Tensor] - remove comment
    def decode(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        assert isinstance(boxes, torch.Tensor)
        assert isinstance(rel_codes, torch.Tensor)
        # boxes_per_frame = [b.size(0) for b in boxes]
        # print("BOXES PER FRAME: ", boxes_per_frame)
        # concat_boxes = torch.cat(boxes, dim=0)
        # print("CONCAT BOXES: ", concat_boxes.shape)
        # TODO delete
        # box_sum = 0
        # for val in boxes_per_frame:
        #     box_sum += val
        # if box_sum > 0:
        #     rel_codes = rel_codes.reshape(box_sum, -1)
        # pred_boxes = self.decode_single(rel_codes, concat_boxes)
        # if box_sum > 0:
        #     # TODO change output dimension
        #     pred_boxes = pred_boxes.reshape(box_sum, -1, 9)
        pred_boxes = self.decode_single(rel_codes, boxes)
        return pred_boxes

    def decode_single(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.
        Args:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        boxes = boxes.to(rel_codes.dtype)

        print("REL CODES: ", rel_codes.shape)
        print("BOXES: ", boxes.shape)

        x = boxes[:, 0, ...]
        y = boxes[:, 1, ...]
        z = boxes[:, 2, ...]
        depths = boxes[:, 3, ...]
        widths = boxes[:, 4, ...]
        heights = boxes[:, 5, ...]
        x_rots = boxes[:, 6, ...]
        y_rots = boxes[:, 7, ...]
        z_rots = boxes[:, 8, ...]
        print("ALL BOXES SHAPES: ", x.shape, y.shape, z.shape, depths.shape,
              widths.shape, heights.shape, x_rots.shape, y_rots.shape, z_rots.shape)

        # TODO fix to match VoxelNet
        # wx, wy, wz, wd, ww, wh, wxrot, wyrot, wzrot = self.weights
        # dx = rel_codes[:, 0::9] / wx
        # dy = rel_codes[:, 1::9] / wy
        # dz = rel_codes[:, 2::9] / wz
        # dd = rel_codes[:, 3::9] / wd
        # dw = rel_codes[:, 4::9] / ww
        # dh = rel_codes[:, 5::9] / wh
        # dzrot = rel_codes[:, 6::9] / wzrot
        # dxrot = rel_codes[:, 7::9] / wxrot
        # dyrot = rel_codes[:, 8::9] / wyrot
        wx, wy, wz, wd, ww, wh, wxrot, wyrot, wzrot = self.weights
        dx = rel_codes[:, 0, ...] / wx
        dy = rel_codes[:, 1, ...] / wy
        dz = rel_codes[:, 2, ...] / wz
        dd = rel_codes[:, 3, ...] / wd
        dw = rel_codes[:, 4, ...] / ww
        dh = rel_codes[:, 5, ...] / wh
        dzrot = rel_codes[:, 6, ...] / wzrot
        dxrot = rel_codes[:, 7, ...] / wxrot
        dyrot = rel_codes[:, 8, ...] / wyrot
        print("ALL D SHAPES: ", dx.shape, dy.shape, dz.shape, dd.shape,
              dw.shape, dh.shape, dzrot.shape, dxrot.shape, dyrot.shape)

        # Prevent sending too large values into torch.exp()
        dd = torch.clamp(dd, max=self.bbox_xform_clip)
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        diagonals = torch.sqrt(depths * depths + widths * widths)
        pred_ctr_x = dx * diagonals + x
        pred_ctr_y = dy * diagonals + y
        pred_ctr_z = dz * heights + z
        pred_d = torch.exp(dd) * depths
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        pred_xrot = dxrot + x_rots
        pred_yrot = dyrot + y_rots
        pred_zrot = dzrot + z_rots

        pred_boxes = torch.stack(
            (pred_ctr_x, pred_ctr_y, pred_ctr_z, pred_d, pred_w, pred_h, pred_xrot, pred_yrot, pred_zrot), dim=1)
        print("PRED_BOXES SHAPE: ", pred_boxes.shape)
        return pred_boxes


class Matcher:
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.
    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.
    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        "BELOW_LOW_THRESHOLD": int,
        "BETWEEN_THRESHOLDS": int,
    }

    def __init__(self, high_threshold: float, low_threshold: float, allow_low_quality_matches: bool = False) -> None:
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """

        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix: Tensor) -> Tensor:
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.
        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the frames during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the frames during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction

        matched_vals, matches = match_quality_matrix.max(dim=0)

        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None  # type: ignore[assignment]

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold)
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(
                matches, all_matches, match_quality_matrix)

        print("MATCHES: ", matches)

        return matches

    def set_low_quality_matches_(self, matches: Tensor, all_matches: Tensor, match_quality_matrix: Tensor) -> None:
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        torch.set_printoptions(profile="full")
        print("HIGHEST_QUALITY_FOREACH_GT: ", highest_quality_foreach_gt)
        torch.set_printoptions(profile="default")
        # Find highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.where(
            match_quality_matrix == highest_quality_foreach_gt[:, None])
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


def overwrite_eps(model: nn.Module, eps: float) -> None:
    """
    This method overwrites the default eps values of all the
    FrozenBatchNorm2d layers of the model with the provided value.
    This is necessary to address the BC-breaking change introduced
    by the bug-fix at pytorch/vision#2933. The overwrite is applied
    only when the pretrained weights are loaded to maintain compatibility
    with previous versions.
    Args:
        model (nn.Module): The model on which we perform the overwrite.
        eps (float): The new value of eps.
    """
    for module in model.modules():
        if isinstance(module, FrozenBatchNorm2d):
            module.eps = eps


def retrieve_out_channels(model: nn.Module, size: Tuple[int, int]) -> List[int]:
    """
    This method retrieves the number of output channels of a specific model.
    Args:
        model (nn.Module): The model for which we estimate the out_channels.
            It should return a single Tensor or an OrderedDict[Tensor].
        size (Tuple[int, int]): The size (wxh) of the input.
    Returns:
        out_channels (List[int]): A list of the output channels of the model.
    """
    in_training = model.training
    model.eval()

    with torch.no_grad():
        # Use dummy data to retrieve the feature map sizes to avoid hard-coding their values
        device = next(model.parameters()).device
        tmp_img = torch.zeros((1, 3, size[1], size[0]), device=device)
        features = model(tmp_img)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        out_channels = [x.size(1) for x in features.values()]

    if in_training:
        model.train()

    return out_channels


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total],
                         dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    def __init__(self, delimiter="\t", writer=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.writer = writer

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}",
                    "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            scalars = defaultdict()
            for name, meter in self.meters.items():
                scalars[name] = meter.value
            print("Scalars: ", scalars)
            self.writer.add_scalars("losses", scalars, int(
                re.findall(r"\d+", header)[0]) * 60 + i)

            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def collate_fn(batch):
    return tuple(zip(*batch))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables # printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        # print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    # print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
