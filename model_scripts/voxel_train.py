import torch
from torch import nn, Tensor, tensor

import torchvision
from torch.utils.tensorboard import SummaryWriter
from model.voxel_rcnn import VoxelRCNN
from utils.voxel.anchor_utils import AnchorGenerator
from data_types.target import Target
import torch.optim as optim
# from dataset.nuScenes_radar import nuScenesDataset 
from dataset.zadar_radar import ZadarLabsDataset
import utils.voxel._utils as utils
from utils.voxel.engine import train_one_epoch, evaluate
from torchsummary import summary
from modules.voxel_backbone import VoxelBackbone


def voxel_train():
    writer = SummaryWriter()
    backbone = VoxelBackbone()
    backbone.out_channels = 1536

    anchor_generator = AnchorGenerator()

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = ZadarLabsDataset()
    dataset_test = ZadarLabsDataset()

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-30])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-30:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = VoxelRCNN(backbone=backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # print("Writing Graph")
    # with open("sample.json", "r") as f:
    #     sample_input = json.load(f)
    #     images = [tensor(sample_input["Images"][0]),
    #               tensor(sample_input["Images"][1])]
    #     targets = [Target(boxes=tensor(sample_input["Targets"][0]["boxes"]), labels=tensor(sample_input["Targets"][0]["labels"]),
    #                       masks=tensor(sample_input["Targets"][0]["masks"]), image_id=tensor(sample_input["Targets"][0]["image_id"]),
    #                       area=tensor(sample_input["Targets"][0]["area"]), iscrowd=tensor(sample_input["Targets"][0]["iscrowd"])),
    #                Target(boxes=tensor(sample_input["Targets"][1]["boxes"]), labels=tensor(sample_input["Targets"][1]["labels"]),
    #                       masks=tensor(sample_input["Targets"][1]["masks"]), image_id=tensor(sample_input["Targets"][1]["image_id"]),
    #                       area=tensor(sample_input["Targets"][1]["area"]), iscrowd=tensor(sample_input["Targets"][1]["iscrowd"]))]
    # writer.add_graph(model, (images, targets, torch.BoolTensor([True])), use_strict_trace=False)
    # print("Graph Finished")

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, # printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader,
                        device, epoch, print_freq=10, writer=writer)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device, epoch, writer=writer)

    writer.flush()
