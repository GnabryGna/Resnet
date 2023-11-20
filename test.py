import torch
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
            prob = outputs.softmax(dim=1) #all type tensor
            top_1pred = prob.argmax(dim=1) #제일 확류 높은거 class번호
            top_3pred = prob.topk(k=3, dim=1)
            top_1acc = top_1pred.eq(labels.to(device)).float().mean() # top-1 Accuracy
            top_3acc = (top_3pred[1][:, 0] == labels.to(device)).float().mean() # top-3 Accuracy

        print(f"top1_accuracy : {top_1acc.item()}, top-3 accuracy : {top_3acc.item()}")

        #PCA 만들기