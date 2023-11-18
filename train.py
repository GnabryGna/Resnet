import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim

import build_model

def train(dataset):
    loss_ = []
    batch_size = len(dataset)
    learning_rate = 0.01
    model = build_model.ResNet().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(10):
        running_loss = 0
        for data in dataset:
            inputs, target = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_func(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss_.append(running_loss / batch_size)
        print("[{}] loss : {}".format(epoch + 1, running_loss / batch_size))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(123)
    if device == 'cuda':
        torch.cuda.manual_seed_all(123)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
    # print(type(trainloader))
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
   #print(type(trainloader))
    train(trainloader)
