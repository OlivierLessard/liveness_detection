import torch
import statistics
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
# from model import Generator as Model3D
import argparse
from dataloaders.data_7_EfficientNet import CsvDataset
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
 
    test_csv_dir = './face_db/testing.csv'

    # Load data
    transform1 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    testing_dataset = CsvDataset(csv_file=test_csv_dir, transform1=transform1, should_invert=False)
    test_loader = DataLoader(testing_dataset, shuffle=True, num_workers=4 * torch.cuda.device_count(), batch_size=batch_size, pin_memory=True)
    
    print('test_loader', len(test_loader))

    is_use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda" if is_use_cuda else "cpu")
    device = torch.device("cpu")
    print("Run the script on Device: ", device)

    #################
    # model_effNet = trained_models.efficientnet_b7(pretrained=False).to(device)
    # model = Discriminator().to(device)
    from train_CNN import get_mobilenet
    mobilenet = get_mobilenet().to(device)
    path = 'trained_models/mobilenet_v2.ckpt'


    checkpoint = torch.load(path, map_location=device)
    model_loaded_state = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    # model.load_state_dict(model_loaded_state)
    mobilenet.load_state_dict(model_loaded_state)

    # model_effNet_loaded_state = {k.replace('module.', ''): v for k, v in checkpoint['model3D_state_dict'].items()}
    # model_effNet.load_state_dict(model_effNet_loaded_state)
    # model.eval()
    mobilenet.eval()
    # model_effNet.eval()
    
   ################
    accuracy_list = []    
    # testing_data_length = [10,15,20,25,30,35,40,45]
    # testing_data_length = [10]
    testing_data_length = [21]
    for testing_length in testing_data_length:
        accuracy_init_list = []
        for epoch in range(1):
            get_corrects = 0.0
            for i, data in enumerate(test_loader):
                # print(i)
                if i == testing_length :
                    break
                # Prepare sample and target
                img1, img2, label_image_1, label_image_2 = data  # label_0 is the label of image_0 and label_1 is for image_1
                img1, img2, label_image_1, label_image_2 = img1.to(device), img2.to(device), label_image_1.to(device), label_image_2.to(device)
                
                # latent_1 = model_effNet(img1)
                # latent_2 = model_effNet(img2)
                y1o_softmax = mobilenet(img1)
                y2o_softmax = mobilenet(img2)

                # _, y1o_softmax = model(latent_1)
                # _, y2o_softmax = model(latent_2)

                get_corrects += torch.max(y1o_softmax, 1)[1] == label_image_1
                get_corrects += torch.max(y2o_softmax, 1)[1] == label_image_2
                # print('get_corrects', get_corrects)
            variable_acc = get_corrects.item() / (testing_length*2)
            accuracy_init_list.append(variable_acc)
        print('Accuracy@' + str(testing_length) + ': ', str(statistics.mean(accuracy_init_list)))
        accuracy_list.append(statistics.mean(accuracy_init_list))
    print('accuracy_list', accuracy_list)

    # confusion matrix
    from torchvision.datasets import ImageFolder
    test_set = ImageFolder(root='face_db/testing', transform=transform1)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    y_true = []
    y_pred = []
    for i, data in enumerate(test_loader):
        # Prepare sample and target
        img1, label_image_1 = data  # label_0 is the label of image_0 and label_1 is for image_1
        img1, label_image_1 = img1.to(device), label_image_1.to(device)

        # latent_1 = model_effNet(img1)
        # latent_2 = model_effNet(img2)

        # _, y1o_softmax = model(latent_1)
        # _, y2o_softmax = model(latent_2)
        y1o_softmax = mobilenet(img1)

        get_corrects += torch.max(y1o_softmax, 1)[1] == label_image_1
        # print('get_corrects', get_corrects)

        y_true.extend([label_image_1.item()])
        y_pred.extend([torch.max(y1o_softmax, 1)[1].item()])

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
    plt.savefig(os.path.join('trained_models', 'confusion_matrix_v2.png'))



if __name__ == '__main__':
    main()