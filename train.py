
# 准备数据集
import torch
import torchvision
from torch import nn
from torch.nn import Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=torchvision.transforms.ToTensor())

# 数据长度
print("训练集长度为：{}".format(len(train_data)))
print("测试集长度为：{}".format(len(test_data)))

# 加载数据
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# 搭建模型
class Demo(nn.Module):
    def __init__(self):
        super(Demo, self).__init__()
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
        x = self.model1(x)
        return x

model = Demo()
model.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)
# 优化器
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 设置训练的参数
#记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练的论数
epoch = 10
# 添加tensorboard
writer =SummaryWriter("logs_train")


for i in range(epoch):
    print("------第{}轮训练开始------".format(i + 1))

    # 训练步骤开始
    model.train()  # 当网络中有Dropout，BatchNorm层时，调用model.train()才有用
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = model(imgs)
        loss = loss_fn(output, targets)

        # 优化器模型
        optimizer.zero_grad() # 梯度归零
        loss.backward() # 反向传播，求得参数
        optimizer.step() # 更新参数

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数:{}，Loss:{}".format(total_train_step,loss.item()))# .item():将tensor类型的数转为python的数值
            writer.add_scalar("train loss",loss.item(),total_train_step)


    # 测试步骤开始
    # train.eval() # # 当网络中有Dropout，BatchNorm层时，调用model.eval()才有用
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad(): # 无梯度，保证模型不被更新
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss += loss.item()
            accuarcy = (outputs.argmax(1)==targets).sum()
            total_accuracy+=accuarcy

        print("整体测试集上的Loss:{}".format(total_test_loss))
        print("整体测试集上的准确率:{}".format(total_accuracy/len(test_data)))
        writer.add_scalar("test loss",total_test_loss,total_test_step)
        writer.add_scalar("test_accuracy",total_accuracy/len(test_data),total_test_step)
        total_test_step += 1

        if(total_test_step==6):
            torch.save(model,"../model/model_{}.pth".format(i))
            print("模型已保存")

writer.close()


