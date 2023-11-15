import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(123)
if device == 'cuda':
    torch.cuda.manual_seed_all(123)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=0)
#print(type(trainloader))
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck') #data set 끝


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, **kwargs) -> None:
        super(Conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)  # kernel size = ...
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        if not self.activation:
            return self.batchnorm(self.conv(x))
        return self.relu(self.batchnorm(self.conv(x))


class Res_block(nn.Module):
    def __init__(self, in_channels, red_channels, out_channels, is_plain=False):
        super(Res_block, self).__init__()
        self.relu = nn.ReLU()
        self.is_plain = is_plain

        if in_channels == 64:
            self.convseq = nn.Sequential(
                Conv_block(in_channels, red_channels, kernel_size=1, padding=0),
                Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
                Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
            )
            self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        elif in_channels == out_channels:
            self.convseq = nn.Sequential(
                Conv_block(in_channels, red_channels, kernel_size=1, padding=0),
                Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
                Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)
            )
            self.iden = nn.Identity()
        else:
            self.convseq = nn.Sequential(
                Conv_block(in_channels, red_channels, kernel_size=1, padding=0, stride=2),
                Conv_block(red_channels, red_channels, kernel_size=3, padding=1),
                Conv_block(red_channels, out_channels, activation=False, kernel_size=1, padding=0)

            )
            self.iden = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)

    def forward(self, x):
        y = self.convseq(x)
        if self.is_plain:
            x = y
        else:
            x = y + self.iden(x)
        x = self.relu(x)  # relu(skip connection)
        return x

