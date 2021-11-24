import torch
from torch.utils.tensorboard import SummaryWriter
from model import Model


writer = SummaryWriter()
model = Model()
print(model)

input = torch.randn(1, 1, 32, 32)
output = model(input)
print(output)

writer.add_graph(model, input)
writer.close()