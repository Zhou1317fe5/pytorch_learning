import torch
import torchvision.datasets
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False) # vgg16模型，未训练。默认参数
vgg16_true = torchvision.models.vgg16(pretrained=True)  # vgg16模型，预训练。

print(vgg16_true) # 有1000个类别

# 将vgg16模型运用于CIFAR10数据集
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# 添加线性层
vgg16_true.classifier.add_module("add_linear",nn.Linear(1000,10))
print(vgg16_true)

# 修改模型
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096,10)
print(vgg16_false)