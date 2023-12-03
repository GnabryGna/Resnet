import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.residual_function = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),

            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels)
        )

        if self.stride != 1 or self.in_channels != self.out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.out_channels)
            )
        self.seblock = SEBlock(out_channels)
    def forward(self, x):
        out = self.residual_function(x)
        if self.stride != 1 or self.in_channels != self.out_channels:
            x = self.downsample(x)
        x = self.seblock(x) * x
        out = F.relu(x + out)
        return out


class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super(SEBlock, self).__init__()

        # 본문 2번항목: Squeeze Operation
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        # 본문 3번 항목: Excitation Operation
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x)  # Global Average Pooling
        x = x.view(x.size(0), -1)  # Batch size축은 놔두고 나머지를 일렬로 쭉 펴기
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)  # 원래대로 복구
        return x

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = self.make_layer(64, 3, stride=1) #첫 스트파이드
        self.layer2 = self.make_layer(128, 4, stride=2)
        self.layer3 = self.make_layer(256, 6, stride=2)
        self.layer4 = self.make_layer(512, 3, stride=2)

        self.gap = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1) #block 수랑 같음.
        # print(strides)
        layers = []

        for stride in strides: #stride수 만큼 block을 만드네?
            block = ResidualBlock(self.in_channels, out_channels, stride)
            layers.append(block)
            # print(layers)
            self.in_channels = out_channels
            # print(nn.Sequential(*layers))
        return nn.Sequential(*layers)

    def forward(self, x):# ([batchsize, feature_channel, height, width])
        out = self.base(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1) #FL 을 지나기 위해 shape 변경 1차원으로 이게 avg pooling일듯?
        out = self.fc(out)
        return out