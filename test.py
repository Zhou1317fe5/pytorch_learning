
import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "./data/dog.png"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

class Demo(nn.Module):
    def __init__(self):
        super(Demo, self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

model = torch.load("../model/model_5.pth", map_location=torch.device('cpu')) # map_location:如果模型是在GPU上训练的，本机只有CPU，则需要将模型映射到CPU上
print(model)

image = torch.reshape(image, (1, 3, 32, 32)) # 需要batch_size,将图片reshape
model.eval() # 将模型转为测试模式
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))
