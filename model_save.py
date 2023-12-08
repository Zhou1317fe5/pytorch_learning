import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1,模型结构+模型参数
torch.save(vgg16, "../model/vgg16_method1.pth")

# 保存方式2，模型参数（官方推荐）
torch.save(vgg16.state_dict(), "../model/vgg16_method2.pth")

# 方式1陷阱
class Demo(nn.Module):
    def __init__(self):
        super(Demo, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

demo = Demo()
torch.save(demo, "../model/demo_method1.pth")