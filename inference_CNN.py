import torch
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
# from model import Generator as Model3D
import argparse
import seaborn as sn
import os.path
import matplotlib.pyplot as plt
import pandas as pd


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Fraud Detection in Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--gpu_index', type=int, default=0, help='Index of the GPU device to use')
    args = parser.parse_args()
    batch_size = args.bs

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    print("Run the script on Device: ", device)

    from train_baseline_1 import get_densenet
    model = get_densenet().to(device)
    path = 'trained_models_for_paper/scheme1_densenet121_baseline.ckpt'

    checkpoint = torch.load(path, map_location=device)
    model_loaded_state = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(model_loaded_state)
    model.eval()

    # Load data
    from torchvision.datasets import ImageFolder
    transform1 = transforms.Compose([transforms.Resize((240, 240)), transforms.ToTensor()])
    test_set = ImageFolder(root='face_db/testing', transform=transform1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    y_true = []
    y_pred = []
    image_names = []
    image_pred = []
    for i, data in enumerate(test_loader):
        # Prepare sample and target
        img1, label_image_1 = data  # label_0 is the label of image_0 and label_1 is for image_1
        img1, label_image_1 = img1.to(device), label_image_1.to(device)

        y1o_softmax = model(img1)

        y_true.extend([label_image_1.item()])
        y_pred.extend([torch.max(y1o_softmax, 1)[1].item()])

        image_names.append(test_loader.dataset.imgs[i])
        image_pred.append(torch.max(y1o_softmax, 1)[1].item())
        print(test_loader.dataset.imgs[i], torch.max(y1o_softmax, 1)[1].item())

    print(image_names)
    print(image_pred)
    from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

    # Assuming y_true and y_pred are your true and predicted labels, respectively
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("precision:", precision)
    print("f1:", f1)

    # build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    classes = test_set.classes
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, fmt=".2g")
    plt.savefig(os.path.join('trained_models_for_competition', 'confusion_matrix_scheme1_densenet121_baseline.png'))



if __name__ == '__main__':
    main()