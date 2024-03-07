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
from model.gan_model import Discriminator, Generator
from torchvision.datasets import ImageFolder
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Fraud Detection in Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--gpu_index', type=int, default=0, help='Index of the GPU device to use')
    args = parser.parse_args()
    batch_size = args.bs

    # Load data
    transform1 = transforms.Compose([transforms.Resize((240, 240)), transforms.Grayscale(), transforms.ToTensor()])
    test_set = ImageFolder(root='face_db/testing', transform=transform1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Load model
    device = torch.device("cpu")
    print("Run the script on Device: ", device)
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    path = 'trained_models_for_paper/net_G_D.ckpt'
    checkpoint = torch.load(path, map_location=device)
    netG_state = {k.replace('module.', ''): v for k, v in checkpoint['model_G_state_dict'].items()}
    netD_state = {k.replace('module.', ''): v for k, v in checkpoint['model_D_state_dict'].items()}
    netG.load_state_dict(netG_state)
    netD.load_state_dict(netD_state)
    netG.eval()
    netD.eval()

    y_true = []
    y_pred = []
    image_names = []
    image_pred = []
    for i, data in enumerate(test_loader):
        # Prepare sample and target
        img1, label_image_1 = data  # label_0 is the label of image_0 and label_1 is for image_1
        img1, label_image_1 = img1.to(device), label_image_1.to(device)

        # inference
        _, x_mask_1, mask_1 = netG(img1)
        features, y_softmax = netD(x_mask_1)
        pred = torch.max(y_softmax, 1)[1].item()  # v2

        # append results
        y_true.extend([label_image_1.item()])
        y_pred.extend([pred])
        image_names.append(test_loader.dataset.imgs[i])
        image_pred.append(pred)
        print(test_loader.dataset.imgs[i], 'pred: ', pred, "label", label_image_1.item())

    print(image_names)
    print(image_pred)

    # metrics
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("precision:", precision)
    print("f1:", f1)

    # confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    classes = test_set.classes
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True, fmt=".2g")
    # plt.savefig(os.path.join('./trained_models_for_competition', 'confusion_matrix_ensemble_v2_v4.png'))
    plt.savefig(os.path.join('trained_models_for_paper', 'confusion_matrix_testing_G_D.png'))


if __name__ == '__main__':
    main()
