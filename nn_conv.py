

"""Conv2d
CLASS torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)


Parameters
in_channels (int) – Number of channels in the input image 输入图像的通道数
out_channels (int) – Number of channels produced by the convolution 输出图像的通道数（就是有几个卷积核的意思，卷积核的数量，有几个卷积核 有几个输出通道）
kernel_size (int or tuple) – Size of the convolving kernel 卷积核的大小
stride (int or tuple, optional) – Stride of the convolution. Default: 1 卷积核横向和纵向步进的大小
padding (int, tuple or str, optional) – Padding added to all four sides of the input. Default: 0 对图像边缘进行填充
 -------------------------------------------
padding_mode (str, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros' 填充模式
dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1 卷积过程中核之间的距离，空洞卷积
groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1 
bias (bool, optional) – If True, adds a learnable bias to the output. Default: True 是否偏置

"""

import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

class convDemo(nn.Module):
    def __init__(self):
        super(convDemo, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3,stride=1, padding=0)

    def forward(self, x):
        x=self.conv1(x)
        return x
    
convDemo = convDemo()

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = convDemo(imgs)
    print(imgs.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("input", imgs, step)

    # torch.Size([64, 6, 30, 30])-[xxx,3,30,30]

    output=torch.reshape(output,(-1,3,30,30)) # 因为tensboard只能输入3通道的图像，所以需要把6通道变成3通道，将多于的通道放入第一个参数中（）设置成-1。https://www.bilibili.com/video/BV1hE411t7RN?t=1372.0&p=18
    writer.add_images("outout",output,step)

    step+=1

writer.close()