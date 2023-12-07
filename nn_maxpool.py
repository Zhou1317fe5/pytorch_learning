"""最大池化
目的：保持输入的特征并减少数据量 
卷积的作用是提取特征，池化的作用是降低特征的数据量



CLASS torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

Parameters
kernel_size (Union[int, Tuple[int, int]]) – the size of the window to take a max over 
stride (Union[int, Tuple[int, int]]) – the stride of the window. Default value is kernel_size
padding (Union[int, Tuple[int, int]]) – Implicit negative infinity padding to be added on both sides
dilation (Union[int, Tuple[int, int]]) – a parameter that controls the stride of elements in the window 空洞卷积
return_indices (bool) – if True, will return the max indices along with the outputs. Useful for torch.nn.MaxUnpool2d later
ceil_mode (bool) – when True, will use ceil instead of floor to compute the output shape 

"""

import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter



dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1], 
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0,1, 1]
                      ],dtype=torch.float32)
input = torch.reshape(input,(-1,1,5,5))
#print(input.shape)

class Demo(nn.Module):
    def __init__(self):
        super(Demo, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.pool(input)
        return output

demo = Demo()
#output = demo(input)
#print(output)

writer = SummaryWriter("logs")
step=0  
for data in dataloader:
    imgs,targets = data
    writer.add_images("input",imgs,step)

    output = demo(imgs)
    writer.add_images("output",output,step)

    step += 1

writer.close()

