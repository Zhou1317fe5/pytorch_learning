import torchvision
import torch
dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)