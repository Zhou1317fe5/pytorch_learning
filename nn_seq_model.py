import torch
from torch import nn
from torch.nn import Sequential
from torch.utils.tensorboard import SummaryWriter


class Demo1(nn.Module):
    def __init__(self):
        super(Demo1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


demo1 = Demo1()
print(demo1)
input = torch.ones((64, 3, 32, 32))
output1 = demo1(input)
print(output1.shape)


class Demo2(nn.Module):
    def __init__(self):
        super(Demo2, self).__init__()
        self.model1 = Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x=self.model1(x)
        return x

demo2 = Demo2()
print(demo2)
output2 = demo2(input)
print(output2.shape)


writer = SummaryWriter('./logs_seq')
writer.add_graph(demo2, input)
writer.close()