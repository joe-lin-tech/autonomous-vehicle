import utils.transformer._utils as utils

def train_one_epochs(model, optimizer, data_loader, device, epoch, print_freq, scaler=None, writer=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ", writer=writer)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        # TODO convert back when upgrading torch to 1.10.0
        # lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        # )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=warmup_iters, eta_min=0)

    for pointclouds, targets in metric_logger.log_every(data_loader, print_freq, header):
        processed_pointclouds = []
        for pc in list(pointclouds):
            if len(pc) < 128:
                processed_pointclouds.append(torch.vstack([pc, torch.zeros((128 - len(pc), 5))]).unsqueeze(0))
            else:
                processed_pointclouds.append(torch.unsqueeze(pc[:128], 0))

        pointclouds = torch.cat(processed_pointclouds, dim=0)

# TODO rewrite file



import numpy as np
from configs.config import T
from torch import tensor
from data_types.target import VoxelTarget
import utils.transformer._utils as utils
import torch
import time
import sys
import math

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None, writer=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ", writer=writer)
    metric_logger.add_meter("lr", utils.SmoothedValue(
        window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        # TODO convert back when upgrading torch to 1.10.0
        # lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        #     optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        # )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=warmup_iters, eta_min=0)

    # print("Initial Metric Logger: ", metric_logger)

    for frames, targets in metric_logger.log_every(data_loader, print_freq, header):
        voxel_features = [frame for frame in frames]
        processed_voxel_features = []
        for i in range(len(voxel_features)):
            if len(voxel_features[i]) < 128:
                processed_voxel_features.append(tensor(np.vstack([np.array(
                    voxel_features[i]), np.zeros((128 - len(voxel_features[i]), 5))])).unsqueeze(0))
            else:
                processed_voxel_features.append(
                    tensor(np.array(voxel_features[i][:128])).unsqueeze(0))
        # voxel_coords = [frame[1] for frame in frames]
        # max_points = max(list(map(len, voxel_features)))
        # processed_voxel_features = []
        # processed_voxel_coords = []
        # for voxel_feature, voxel_coord in zip(voxel_features, voxel_coords):
        #     if len(voxel_feature) < max_points:
        #         processed_voxel_features.append(tensor(np.vstack([np.array(
        #             voxel_feature), np.zeros((max_points - len(voxel_feature), T, 7))])).unsqueeze(0))
        #         processed_voxel_coords.append(tensor(np.vstack([np.array(voxel_coord), np.zeros(
        #             (max_points - len(voxel_coord), 3))])).unsqueeze(0))
        #     else:
        #         processed_voxel_features.append(
        #             tensor(voxel_feature).unsqueeze(0))
        #         processed_voxel_coords.append(tensor(voxel_coord).unsqueeze(0))
        # frames = torch.cat(processed_voxel_features).float().to(
        #     device), torch.cat(processed_voxel_coords).long().to(device)
        frames = torch.cat(processed_voxel_features).float().to(device)
        targets = [{k: v.to(device) for k, v in t._asdict().items()}
                   for t in targets]
        targets = [VoxelTarget(boxes=t["boxes"], labels=t["labels"],
                               frame_id=t["frame_id"], volume=t["volume"]) for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # TODO remove when done
            targets = [t.boxes for t in targets]
            detections, loss_dict = model(frames, targets)
            losses = sum(loss for loss in loss_dict.values())
        detection_boxes, detection_scores = detections
        print("CONVERTED DETECTIONS...")

        # TODO fix write to data.json
        # with open("display/data/" + str(datetime.datetime.now()) + ".json", "w") as f:
        #     json_targets = [t.tensor_to_list() for t in targets]
        #     json_boxes = [[box.tolist() for box in boxes]
        #                   for boxes in detection_boxes]
        #     json_scores = [[score.tolist() for score in scores]
        #                    for scores in detection_scores]
        #     json_losses = [loss.clone().detach().tolist()
        #                    for loss in loss_dict.values()]
        #     json.dump(dict(targets=json_targets,
        #               boxes=json_boxes, scores=json_scores, losses=json_losses), f)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        print("REDUCED LOSSES...")

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            # print(f"Loss is {loss_value}, stopping training")
            # print(loss_dict_reduced)
            sys.exit(1)
        print("START OPTIMIZER")
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            print("START BACKWARD")
            losses.backward()
            print("FINISH BACKWARD")
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    print("Final Metric Logger: ", metric_logger)

    return metric_logger


# TODO fix evaluation method
@torch.inference_mode()
def evaluate(model, data_loader, device, epoch, writer=None):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ", writer=writer)
    header = f"Test: [{epoch}]"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()}
                   for t in outputs]
        model_time = time.time() - model_time

        res = {target.image_id.item(): output for target,
               output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time,
                             evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
