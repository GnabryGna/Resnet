import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
import resnet
import alexnet
from tqdm import tqdm
from datetime import datetime
import os
import pandas as pd
import test
import wandb
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(123)
if device == 'cuda':
    torch.cuda.manual_seed_all(123)
transform = transforms.Compose(
    [
     transforms.Resize((256, 256)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 64
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)#50000data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) #10000
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

def train(dataset, model, model_name):
    lossArray = []
    model_path = './model/'
    model_list = os.listdir(model_path)
    learning_rate = 0.01
    epoch = 10
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    args = {
        "learning_rate": learning_rate,
        "epochs": epoch,
        "batch_size": batch_size
    }
    wandb.config.update(args)
    for _ in tqdm(range(epoch)):
        running_loss = 0
        for idx, (images, labels) in tqdm(enumerate(dataset)): #64*768data batch = 64
            inputs, target = images.float().to(device), labels.long().to(device)
            print(inputs.size())
            # print(inputs.size()) # 64, 3, 32, 32
            optimizer.zero_grad()
            outputs = model(inputs) #Inputs size = 64, 3, 32, 32
            #print(outputs.size())
            loss = loss_func(outputs, target.detach())
            loss.backward()
            optimizer.step()
            prob = outputs.softmax(dim=1) #확률 softmax
            pred = prob.argmax(dim=1) #predict
            acc = pred.eq(labels.to(device)).float().mean()
            # print("loss.item() = ", loss.item())
            running_loss += loss.item()
            # print("runningLoss", running_loss)
            wandb.log({"loss": loss.item()})
            if (idx + 1) % 64 == 0:
                print("[{}] loss : {}, Accuracy : {}".format(idx + 1, running_loss/64, acc.item()))
                running_loss = 0

                # lossArray.append(loss.item())
    day = datetime.now()
    torch.save(model.state_dict(), f'./model/{day.month}{day.day}_{model_name}_epoch{epoch}_Adam_0.02') #model 저장
    wandb.finish()
    # lossArray = np.array(lossArray) #loss graph 생성
    # df = pd.DataFrame(lossArray)
    # df.to_csv(f'.//loss.csv')


# model = resnet.ResNet().to(device)
# wandb.init(name="ResNet Loss", project="ResNet exp", entity="rlarmsgk2")
# model_name = 'resnet'
# train(trainloader, model, model_name)
# test.test(testloader, model)
model = alexnet.AlexNet().to(device)
wandb.init(name="AlexNet Loss", project="AlexNet exp", entity="rlarmsgk2")
model_name = 'alexnet'
train(trainloader, model, model_name)
# test.test(testloader, model)
