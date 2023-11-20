import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import resnet
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(123)
if device == 'cuda':
    torch.cuda.manual_seed_all(123)

def test(testdata):
    with torch.no_grad():
        acc = 0
        model_path = './model/'
        model_list = os.listdir(model_path)
        if model_list:
            file_paths_with_time = [(os.path.join(model_path, file), os.path.getmtime(os.path.join(model_path, file)))
                                    for file in model_list]
            sorted_file_paths = sorted(file_paths_with_time, key=lambda x: x[1], reverse=True)
            latest_model = sorted_file_paths[0][0]
            model = resnet.ResNet().to(device)
            model.load_state_dict(torch.load(f'{latest_model}'))
            model.eval()


        for idx, (images, labels) in enumerate(testdata):
            images, labels = images.float().to(device), labels.long().to(device)
            outputs = model(images)
            loss = F.cross_entropy(input=outputs, target=labels)
            prob = outputs.softmax(dim=1)
            pred = prob.argmax(dim=1)
            acc = pred.eq(labels.to(device)).float().mean()
            print(pred.eq(labels.to(device)).float())
        print(f"test_accuracy : {acc.item()}")

        #PCA 만들기