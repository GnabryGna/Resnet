from torch import nn
import torch
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64,192,3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=(256*3*3), out_features=1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=num_classes),
        )

    # def init_bias(self):
    #     for layer in self.net:
    #         if isinstance(layer, nn.Conv2d):
    #             nn.init.normal_(layer.weight, mean=0, std=0.01)
    #             nn.init.constant_(layer.bias, 0)
    #     nn.init.constant_(self.net[4].bias, 1)
    #     nn.init.constant_(self.net[10].bias, 1)
    #     nn.init.constant_(self.net[12].bias, 1)

    def forward(self,x):
        x = self.net(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
