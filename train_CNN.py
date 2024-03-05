import argparse
import os
import torch
import csv
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from loss import ContrastiveLoss
from dataloaders.data_7_EfficientNet import CsvDataset
from torch.optim import lr_scheduler
import torchvision.models as models
from utils_EfficientNety import AverageMeter


def get_mobilenet():
    """
    Load the model. Define the architecture to use in the config.py file.

    :return: the model
    """
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[-1] = nn.Linear(1024, 2)
    return model


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Check Quality of the Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--gpu_index', type=int, default=0, help='Index of the GPU device to use')
    parser.add_argument('--array_param', type=str, default='1-4', help='Array parameter')

    args = parser.parse_args()
    best_acc = 0.0
    batch_size = args.bs

    torch.cuda.empty_cache()
    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")
    print('--------- device:', device, args.gpu_index)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    # dataloader
    dataset_sizes, train_loader = get_pair_training_dataloader(batch_size)

    # model
    mobilenet = get_mobilenet().to(device)
    print('Model created.')

    # Training parameters
    optimizer = torch.optim.Adam(mobilenet.parameters(), args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Loss
    criterion_contrastive = ContrastiveLoss()
    loss_cross = torch.nn.CrossEntropyLoss()


    print("Total number of batches in train loader are :", len(train_loader))

    # Start training...
    for epoch in range(args.epochs):

        losses = AverageMeter()

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Switch to train mode
        mobilenet.train()

        total_batch_loss = 0.0
        get_corrects = 0.0

        for i, data in enumerate(train_loader):
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()

            # Prepare sample and target
            image_1, image_2, label_1, label_2 = data
            image_1, image_2, label_1, label_2 = image_1.to(device), image_2.to(device), label_1.to(device), label_2.to(device)

            predicted_probabilities_1 = mobilenet(image_1)
            predicted_probabilities_2 = mobilenet(image_2)
            correct_predictions_1 = (predicted_probabilities_1.argmax(1) == label_1).type(torch.float)
            correct_predictions_2 = (predicted_probabilities_2.argmax(1) == label_2).type(torch.float)
            get_corrects += torch.sum(torch.logical_and(correct_predictions_2, correct_predictions_1))

            loss_cross_entropy_1 = loss_cross(predicted_probabilities_1, label_1)
            loss_cross_entropy_2 = loss_cross(predicted_probabilities_2, label_2)

            softmax = torch.nn.Softmax(dim=1)
            prob2 = softmax(predicted_probabilities_2)
            loss_total = (torch.mean(prob2, dim=0)[0] * loss_cross_entropy_1 +
                          torch.mean(prob2, dim=0)[1] * loss_cross_entropy_2)  # + contrastive_loss

            loss_total.backward()
            optimizer.step()

            # Update step
            total_batch_loss = loss_total
            losses.update(total_batch_loss.data.item(), image_1.size(0))

            torch.cuda.empty_cache()
        
        variable_acc = get_corrects.item() / dataset_sizes 

        # Log progress; print after every epochs into the console
        print('Epoch: [{:.4f}] \t The loss of this epoch is: {:.4f} \t The accuracy of this epoch is: {:.4f} '.format(epoch, losses.avg, variable_acc))

        if not os.path.isdir('trained_models'):
            os.makedirs('trained_models')
        if variable_acc > best_acc:
            best_acc = save_model(best_acc, mobilenet, variable_acc)

        # save the losses avg in .csv file
        if not os.path.isdir('./loss_1'):
            os.makedirs('./loss_1')
        with open('./loss_1/' + "loss_mobilenet_v4.csv", 'a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, losses.avg, variable_acc])
           
        scheduler.step()


def save_model(best_acc, mobilenet, variable_acc):
    print("Here the training accuracy got reduced, hence printing")
    print('Current best epoch accuracy is {:.4f}'.format(variable_acc), 'previous best was {}'.format(best_acc))
    best_acc = variable_acc
    torch.save({
        # 'model_state_dict': model.state_dict(),
        # 'model3D_state_dict': model_effNet.state_dict()},
        'model_state_dict': mobilenet.state_dict(),
        'model3D_state_dict': mobilenet.state_dict()},
        'trained_models/mobilenet_v4.ckpt')
    return best_acc


def get_pair_training_dataloader(batch_size):
    train_csv_dir = './face_db/training' + '.csv'
    transform1 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    training_dataset = CsvDataset(csv_file=train_csv_dir, transform1=transform1, should_invert=False)
    train_loader = DataLoader(training_dataset, shuffle=True, num_workers=0,
                              batch_size=batch_size, pin_memory=True)  # num_workers=4 * torch.cuda.device_count()
    dataset_sizes = len(train_loader.dataset)
    return dataset_sizes, train_loader


if __name__ == '__main__':
    main()
