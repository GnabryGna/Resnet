import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
import resnet
from tqdm import tqdm
from datetime import datetime
import os
import pandas as pd
import test
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(123)
if device == 'cuda':
    torch.cuda.manual_seed_all(123)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
#DataLoader 객체 확인
# for index, (images, labels) in enumerate(trainloader): # ([batchsize, channel, height, width])
#     print(f"{index}/{len(trainloader)}", end=' ')
#     print("x shape:", images.shape, end=' ')
#     print("y shape:", labels.shape)
def train(dataset):
    lossArray = []
    model_path = './model/'
    model_list = os.listdir(model_path)
    learning_rate = 0.01
    model = resnet.ResNet().to(device)
    if model_list:
        file_paths_with_time = [(os.path.join(model_path, file), os.path.getmtime(os.path.join(model_path, file))) for file in model_list]
        sorted_file_paths = sorted(file_paths_with_time, key=lambda x: x[1], reverse=True)
        latest_model = sorted_file_paths[0][0]
        model = torch.load(latest_model).to(device)

    # summary(model,(3,32,32), device=device)
    # print(model)
    epoch = 10
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for _ in tqdm(range(epoch)):
        running_loss = 0
        for idx, (images, labels) in tqdm(enumerate(dataset)):
            inputs, target = images.float().to(device), labels.long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, target.detach())
            loss.backward()
            optimizer.step()
            prob = outputs.softmax(dim=9) #확률 softmax
            pred = prob.argmax(dim=1) #predict
            acc = pred.eq(labels.to(device)).float().mean()
            running_loss += loss.item()
            if (idx + 1) % 128 == 0:
                print("[{}] loss : {}, Accuracy : {}".format(idx + 1, loss.item(), acc.item()))
                lossArray.append(loss.item())
    day = datetime.now()
    torch.save(model.state_dict(), f'./model/{day.month}{day.day}_epoch{epoch}')

    lossArray = np.array(lossArray) #loss graph 생성
    df = pd.DataFrame(lossArray)
    df.to_csv(f'.//loss.csv')

# train(trainloader)
test.test(testloader)