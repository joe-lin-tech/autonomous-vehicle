import torch
from torch import nn, Tensor, tensor

import torchvision
from torch.utils.tensorboard import SummaryWriter
from models.voxel_rcnn import VoxelRCNN
from utils.anchor_utils import AnchorGenerator
from data_types.target import Target
import torch.optim as optim
from dataset.penn_fundan import PennFudanDataset
from utils.transform import get_transform
import utils._utils as utils
from utils.engine import train_one_epoch, evaluate


def main():
    writer = SummaryWriter()

    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

    # backbone = Backbone(1280)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=7, sampling_ratio=2)
    voxel_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=14, sampling_ratio=2)

    # TODO remove testing code
    # model = VoxelRCNN(backbone=backbone, num_classes=2, rpn_anchor_generator=anchor_generator,
    #                   box_roi_pool=roi_pooler, voxel_roi_pool=voxel_roi_pooler)
    # model.eval()
    # input = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]

    # TODO change back
    # targets = [{"boxes": tensor([[1, 2, 3, 4]]), "labels": tensor([1], dtype=torch.int64), "masks": torch.rand(1, 3, 4)}, {
    #     "boxes": tensor([[5, 6, 7, 8]]), "labels": tensor([1], dtype=torch.int64), "masks": torch.rand(1, 7, 8)}]
    # targets = [Target(tensor([[1, 2, 3, 4]]), tensor([1], dtype=torch.int64), torch.rand(
    #     1, 3, 4)), Target(tensor([[5, 6, 7, 8]]), tensor([1], dtype=torch.int64), torch.rand(1, 7, 8))]
    # print("Initial Target Type: ", type(targets[1]))
    # output = model(input, targets)
    # print("Output: ", output)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    print("get_transform: ", get_transform(train=True))
    dataset = PennFudanDataset(get_transform(train=True))
    dataset_test = PennFudanDataset(get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = VoxelRCNN(backbone=backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator,
                      box_roi_pool=roi_pooler, voxel_roi_pool=voxel_roi_pooler)

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

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader,
                        device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            print("Inputs: ", inputs, inputs.shape)
            print("Labels: ", labels, labels.shape)
            # forward + backward + optimize
            loss = model(inputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    model.train()

    # writer.add_graph(model, (input, targets), use_strict_trace=True)
    # writer.close()


if __name__ == '__main__':
    main()
