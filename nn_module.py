from torch import nn
import torch

class Demo(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output=input+1
        return output
    
demo = Demo()
x = torch.tensor(1.0)
output = demo(x)
print(output)
