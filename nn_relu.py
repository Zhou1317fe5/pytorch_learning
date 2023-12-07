import torch
import torch.nn as nn
from torch.nn import ReLU,Sigmoid
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

class Demo(nn.Module):
    def __init__(self):
        super(Demo, self).__init__()
        self.relu1=ReLU()
        self.sigmoid1=Sigmoid()

    def forward(self, input):
        #output=self.relu1(input)
        output=self.sigmoid1(input)
        return output
    
demo = Demo()
writer = SummaryWriter("./logs_relu")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("inout",imgs,global_step=step)

    output = demo(imgs)
    writer.add_images("output",output,global_step=step)
    step += 1
writer.close
