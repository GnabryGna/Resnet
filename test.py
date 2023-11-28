import torch
import os
import resnet
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(123)
if device == 'cuda':
    torch.cuda.manual_seed_all(123)


def test(testdata, model):
    with torch.no_grad():
        acc = 0
        model_path = './model/'
        model_list = os.listdir(model_path)
        if model_list:
            file_paths_with_time = [(os.path.join(model_path, file), os.path.getmtime(os.path.join(model_path, file)))
                                    for file in model_list]
            sorted_file_paths = sorted(file_paths_with_time, key=lambda x: x[1], reverse=True)
            latest_model = sorted_file_paths[0][0]
            model.load_state_dict(torch.load(f'{latest_model}'))
            model.eval()

        for idx, (images, labels) in enumerate(testdata):
            images, labels = images.float().to(device), labels.long().to(device)
            outputs = model(images)
            prob = outputs.softmax(dim=1)  # all type tensor
            top_1pred = prob.argmax(dim=1)  # 제일 확률 높은거 class번호
            top_3pred = prob.topk(k=3, dim=1)
            top_1acc = top_1pred.eq(labels.to(device)).float().mean()  # top-1 Accuracy
            top_3acc = (top_3pred[1][:, 0] == labels.to(device)).float().mean()  # top-3 Accuracy
            # Class별로 argmax값 뽑아서 idx랑 같이 묶어놔. idx가 class 번호임. idx별로 argmax값 나중에 저장해놓고 mean과 variance 저장하면 될듯?
        print(f"top1_accuracy : {top_1acc.item()}, top-3 accuracy : {top_3acc.item()}")

        model = nn.Sequential(*list(model.children())[:-1])
        features = []
        labels = []
        for data, label in testdata:
            # print(data.size())
            output = model(data.to(device))
            features.append(output.squeeze())  # 1인 차원 제거 64x512
            labels.append(label)

        features = torch.cat(features)  # concat
        labels = torch.cat(labels)

        getMeanVar(features, labels) # get mean, var for each classes
        showPCA(features, labels)# show PCA

        # take mean vairance per class

def getMeanVar(features, labels):
    features_list = list(features.cpu())
    labels_size = list(labels.cpu().size())[0]
    labels_list = list(labels.cpu())

    classes_mean = [[], [], [], [], [], [], [], [], [], []]
    classes_var = [[], [], [], [], [], [], [], [], [], []]

    last_mean = []
    last_var = []
    for i in range(labels_size):  # 10000 features mean, variance put it in labels value idx
        classes_mean[labels_list[i].item()].append(features_list[i].float().mean().item())
        classes_var[labels_list[i].item()].append(list(features_list[i]))
    classes_mean = np.array(classes_mean)
    classes_var = np.array(classes_var)
    for i in range(10):
        last_mean.append(classes_mean[i].mean())
        last_var.append(classes_var[i].var())
    print(last_mean, "\n", last_var)



def showPCA(features, labels):
    features = features.detach().cpu().numpy()  # feature vecotr numpy
    # print(features.shape)  # 512 * 10000?
    mean_Vector = np.mean(features, axis=0)
    centered_data = features - mean_Vector
    cov_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    dimensions = 3
    projection_matrix = eigenvectors[:, :dimensions]
    reduced_features = np.dot(centered_data, projection_matrix)

    class_indices = [np.where(labels.numpy() == i)[0] for i in range(10)]
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(10):
        ax.scatter(reduced_features[class_indices[i], 0],
                   reduced_features[class_indices[i], 1],
                   reduced_features[class_indices[i], 2],
                   label=f'Class {i}')
    ax.set_title('PCA on ResNet34 Features for CIFAR-10')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.legend()
    plt.show()
