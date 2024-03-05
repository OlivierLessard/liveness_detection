import torch
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
# from model import Generator as Model3D
import argparse
import seaborn as sn
import os.path
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    import pandas as pd
    df = pd.read_csv("other_results/predictions_benchmarks_original.csv")
    print(df)

    # Arguments
    parser = argparse.ArgumentParser(description='Fraud Detection in Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--gpu_index', type=int, default=0, help='Index of the GPU device to use')
    args = parser.parse_args()
    batch_size = args.bs

    # Load data
    transform1 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    print("Run the script on Device: ", device)


    from train_CNN import get_mobilenet
    mobilenet = get_mobilenet().to(device)
    mobilenet2 = get_mobilenet().to(device)
    mobilenet3 = get_mobilenet().to(device)
    path = 'trained_models/mobilenet_v2.ckpt'
    path2 = 'trained_models/mobilenet_v4.ckpt'
    path3 = 'trained_models/mobilenet_v3.ckpt'

    checkpoint = torch.load(path, map_location=device)
    checkpoint2 = torch.load(path2, map_location=device)
    checkpoint3 = torch.load(path3, map_location=device)
    model_loaded_state = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    model_loaded_state2 = {k.replace('module.', ''): v for k, v in checkpoint2['model_state_dict'].items()}
    model_loaded_state3 = {k.replace('module.', ''): v for k, v in checkpoint3['model_state_dict'].items()}
    mobilenet.load_state_dict(model_loaded_state)
    mobilenet2.load_state_dict(model_loaded_state2)
    mobilenet3.load_state_dict(model_loaded_state3)
    mobilenet.eval()
    mobilenet2.eval()
    mobilenet3.eval()

    # confusion matrix
    from torchvision.datasets import ImageFolder
    test_set = ImageFolder(root='DATASET_VALIDATION', transform=transform1)
    # test_set = ImageFolder(root='face_db/testing', transform=transform1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    y_true = []
    y_pred = []
    image_names = []
    image_pred = []
    for i, data in enumerate(test_loader):
        # Prepare sample and target
        img1, label_image_1 = data  # label_0 is the label of image_0 and label_1 is for image_1
        img1, label_image_1 = img1.to(device), label_image_1.to(device)
        filename = os.path.split(test_loader.dataset.imgs[i][0])[1][:-4]

        y1o_softmax = mobilenet(img1)
        y1o_softmax2 = mobilenet2(img1)
        y1o_softmax3 = mobilenet3(img1)

        pred1 = torch.max(y1o_softmax, 1)[1].item()  # v2
        pred2 = torch.max(y1o_softmax2, 1)[1].item()  # v4
        # pred3 = torch.max(y1o_softmax3, 1)[1].item()  # v3
        pred = int(pred1 and pred2)
        # pred = int(pred1 and pred2 and pred3)

        for ind in df.index:
            # print(df['image'][ind], df['pred'][ind])
            if filename == os.path.split(df['image'][ind])[1][:-4]:
                pred4 = df['pred'][ind]
                break
        pred = int(pred and pred4)

        y_true.extend([label_image_1.item()])
        y_pred.extend([pred])

        image_names.append(test_loader.dataset.imgs[i])
        image_pred.append(pred)
        # print(test_loader.dataset.imgs[i], 'pred1: ', pred1, 'pred2: ', pred2, "perd4", pred4, " so ", pred, "label", label_image_1.item())
        print(test_loader.dataset.imgs[i], 'ours: ', pred, "theirs", pred4, " so ", pred, "label", label_image_1.item())
        # print(test_loader.dataset.imgs[i], 'pred1: ', pred1, 'pred2: ', pred2, 'pred3: ', pred3, " so ", pred)

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
    # plt.savefig(os.path.join('./trained_models', 'confusion_matrix_ensemble_v2_v4.png'))
    plt.savefig(os.path.join('trained_models', 'confusion_matrix_ensemble_v2_v4_depth_AND_final_test.png'))



if __name__ == '__main__':
    main()