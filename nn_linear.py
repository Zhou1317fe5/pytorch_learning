import torchvision
import torch
from torch import nn
from torch.nn import Linear
dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True,drop_last=True)

#img,target= dataloader[0]
#print(img.shape)

class Demo(nn.Module):
    def __init__(self):
        super(Demo,self).__init__()
        self.Linear1=Linear(64*3*32*32,10)

    def forward(self,input):
        output = self.Linear1(input)
        return output


demo = Demo()



for data in dataloader:
    imgs,targets = data
    print(imgs.shape)
    input = torch.flatten(imgs) # 将图片压缩成一维
    print(input.shape)
    output = demo(input)
    print(output.shape)


