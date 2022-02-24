import torch
from collections import defaultdict
import re
import torch.distributed as dist
from collections import defaultdict, deque
import time
import datetime
from configs.config import BATCH_SIZE, D, W


class BoxCoder:
    def __init__(self, weights):
        self.weights = weights
        self.bbox_clip = torch.log(torch.tensor(1000.0 / 16))

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes
        Args:
            reference_boxes (List[Tensor]): reference boxes
            proposals (Tensor): boxes to be encoded
            weights (Tensor[9]): the weights for ``(x, y, z, d, w, h, x_rot, y_rot, z_rot)``

        Returns:
            Tensor: encoded boxes
        """
        num_boxes_per_frame = [D * W // 4 for _ in range(BATCH_SIZE)]
        reference_boxes = torch.cat((reference_boxes), dim=0)
        proposals = torch.flatten(torch.transpose(torch.flatten(
            proposals, start_dim=2), 1, 2), start_dim=0, end_dim=1)

        wx, wy, wz, wd, ww, wh, wx_rot, wy_rot, wz_rot = self.weights

        proposals_x = proposals[:, 0]
        proposals_y = proposals[:, 1]
        proposals_z = proposals[:, 2]
        proposals_d = proposals[:, 3]
        proposals_w = proposals[:, 4]
        proposals_h = proposals[:, 5]
        proposals_x_rot = proposals[:, 6]
        proposals_y_rot = proposals[:, 7]
        proposals_z_rot = proposals[:, 8]

        reference_boxes_x = reference_boxes[:, 0]
        reference_boxes_y = reference_boxes[:, 1]
        reference_boxes_z = reference_boxes[:, 2]
        reference_boxes_d = reference_boxes[:, 3]
        reference_boxes_w = reference_boxes[:, 4]
        reference_boxes_h = reference_boxes[:, 5]
        reference_boxes_x_rot = reference_boxes[:, 6]
        reference_boxes_y_rot = reference_boxes[:, 7]
        reference_boxes_z_rot = reference_boxes[:, 8]

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

        targets = torch.stack(
            (targets_dx, targets_dy, targets_dz, targets_dd, targets_dw, targets_dh, targets_dx_rot, targets_dy_rot, targets_dz_rot), dim=1)
        return torch.split(targets, num_boxes_per_frame, dim=0)

    def decode(self, rel_codes, anchor_boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.
        Args:
            rel_codes (Tensor): encoded boxes
            anchor_boxes (Tensor): anchor boxes.

        Returns:
            Tensor: decoded boxes
        """
        num_boxes_per_frame = [D * W // 4 for _ in range(BATCH_SIZE)]
        rel_codes = torch.flatten(torch.transpose(torch.flatten(
            rel_codes, start_dim=2), 1, 2), start_dim=0, end_dim=1)
        anchor_boxes = torch.flatten(torch.transpose(torch.flatten(
            anchor_boxes, start_dim=2), 1, 2), start_dim=0, end_dim=1)

        wx, wy, wz, wd, ww, wh, wxrot, wyrot, wzrot = self.weights

        anchor_boxes_x = anchor_boxes[:, 0]
        anchor_boxes_y = anchor_boxes[:, 1]
        anchor_boxes_z = anchor_boxes[:, 2]
        anchor_boxes_d = anchor_boxes[:, 3]
        anchor_boxes_w = anchor_boxes[:, 4]
        anchor_boxes_h = anchor_boxes[:, 5]
        anchor_boxes_x_rot = anchor_boxes[:, 6]
        anchor_boxes_y_rot = anchor_boxes[:, 7]
        anchor_boxes_z_rot = anchor_boxes[:, 8]

        rel_codes_dx = rel_codes[:, 0] / wx
        rel_codes_dy = rel_codes[:, 1] / wy
        rel_codes_dz = rel_codes[:, 2] / wz
        rel_codes_dd = rel_codes[:, 3] / wd
        rel_codes_dw = rel_codes[:, 4] / ww
        rel_codes_dh = rel_codes[:, 5] / wh
        rel_codes_dx_rot = rel_codes[:, 6] / wxrot
        rel_codes_dy_rot = rel_codes[:, 7] / wyrot
        rel_codes_dz_rot = rel_codes[:, 8] / wzrot

        # Prevent sending too large values into torch.exp()
        rel_codes_dd = torch.clamp(rel_codes_dd, max=self.bbox_clip)
        rel_codes_dw = torch.clamp(rel_codes_dw, max=self.bbox_clip)
        rel_codes_dh = torch.clamp(rel_codes_dh, max=self.bbox_clip)

        diagonals = torch.sqrt(
            anchor_boxes_d * anchor_boxes_d + anchor_boxes_w * anchor_boxes_w)
        pred_ctr_x = rel_codes_dx * diagonals + anchor_boxes_x
        pred_ctr_y = rel_codes_dy * diagonals + anchor_boxes_y
        pred_ctr_z = rel_codes_dz * anchor_boxes_h + anchor_boxes_z
        pred_d = torch.exp(rel_codes_dd) * anchor_boxes_d
        pred_w = torch.exp(rel_codes_dw) * anchor_boxes_w
        pred_h = torch.exp(rel_codes_dh) * anchor_boxes_h
        pred_xrot = rel_codes_dx_rot + anchor_boxes_x_rot
        pred_yrot = rel_codes_dy_rot + anchor_boxes_y_rot
        pred_zrot = rel_codes_dz_rot + anchor_boxes_z_rot

        pred_boxes = torch.stack(
            (pred_ctr_x, pred_ctr_y, pred_ctr_z, pred_d, pred_w, pred_h, pred_xrot, pred_yrot, pred_zrot), dim=1)
        return torch.split(pred_boxes, num_boxes_per_frame, dim=0)


class Matcher:
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def set_low_quality_matches(self, matches, all_matches, match_quality_matrix):
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = torch.max(match_quality_matrix, 1)
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

    def __call__(self, match_quality_matrix):
        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = torch.max(match_quality_matrix, dim=0)

        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
            matched_vals < self.high_threshold)
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            self.set_low_quality_matches(
                matches, all_matches, match_quality_matrix)
        return matches


class SmoothedValue:
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


def collate_fn(batch):
    return tuple(zip(*batch))


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
