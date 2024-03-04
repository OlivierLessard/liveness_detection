import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from model_classify import Generator as Model3D
from model_attMap import Generator, Discriminator
import argparse
from sklearn.metrics import confusion_matrix
import os.path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

MODEL_DIR = './models/'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Fraud Detection in Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    args = parser.parse_args()
    batch_size = args.bs

    # Load data
    transform1 = transforms.Compose([transforms.Resize((240, 240)), transforms.Grayscale(), transforms.ToTensor()])
    # transform2 = transforms.Compose([transforms.Resize((120, 120)), transforms.ToTensor()])

    device = torch.device("cpu")

    #################
    
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    
    path = 'models/sch1_attMap.ckpt'
    checkpoint = torch.load(path, map_location=device)
    dis_loaded_state = {k.replace('module.', ''): v for k, v in checkpoint['netD_state_dict'].items()}
    netD.load_state_dict(dis_loaded_state)
    gen_loaded_state = {k.replace('module.', ''): v for k, v in checkpoint['netG_state_dict'].items()}
    netG.load_state_dict(gen_loaded_state)
    netD.eval()
    netG.eval()

    from torchvision.datasets import ImageFolder
    test_set = ImageFolder(root='DATASET_VALIDATION', transform=transform1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    y_true = []
    y_pred = []
    image_names = []
    image_pred = []
    for i, data in enumerate(test_loader):
        # Prepare sample and target
        img1, label_image_1 = data  # label_0 is the label of image_0 and label_1 is for image_1
        img1, label_image_1 = img1.to(device), label_image_1.to(device)

        _, x_mask1, mask1 = netG(img1)
        real_vid_feat, y1o_softmax = netD(x_mask1)

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
    plt.savefig(os.path.join('./models', 'confusion_matrix_trained_sch1_attMap_final_test.png'))




if __name__ == '__main__':
    main()
