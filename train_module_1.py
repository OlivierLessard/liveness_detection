# from GPUtil import showUtilization as gpu_usage
# gpu_usage()
import argparse
import os
import torch
import csv
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
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
    
    train_csv_dir = './face_db/training' + '.csv'

    torch.cuda.empty_cache()
    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")
    print('--------- device:', device, args.gpu_index)
    # torch.cuda.set_per_process_memory_fraction(1.0)

    # Create models
    mobilenet = get_mobilenet().to(device)
    # model_effNet = EfficientNet.from_pretrained('efficientnet-b7')
    # model_effNet = models.efficientnet_b7(pretrained=False).to(device)
    # model = Discriminator().to(device)
    print('Model created.')
    
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model.to(device))
    #     model_effNet = nn.DataParallel(model_effNet.to(device))
    
    print('model and cuda mixing done')

    # Training parameters
    optimizer_model = torch.optim.Adam(mobilenet.parameters(), args.lr)
    # optimizer_model = torch.optim.Adam(model.parameters(), args.lr)
    # optimizer_model_effNet = torch.optim.Adam(model_effNet.parameters(), args.lr)
    my_lr_scheduler_model = lr_scheduler.StepLR(optimizer_model, step_size=10, gamma=0.1)
    # my_lr_scheduler_model_effNet = lr_scheduler.StepLR(optimizer_model_effNet, step_size=10, gamma=0.1)

    # Load data
    transform1 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    training_dataset = CsvDataset(csv_file=train_csv_dir, transform1=transform1, should_invert=False)

    train_loader = DataLoader(training_dataset, shuffle=True, num_workers=0,
                              batch_size=batch_size, pin_memory=True) # num_workers=4 * torch.cuda.device_count()

    dataset_sizes = len(train_loader.dataset)

    # Loss
    criterion_contrastive = ContrastiveLoss()
    loss_cross = torch.nn.CrossEntropyLoss()


    print("Total number of batches in train loader are :", len(train_loader))

    # Start training...
    for epoch in range(args.epochs):

        losses = AverageMeter()
        N = len(train_loader)

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Switch to train mode
        mobilenet.train()
        # model.train()
        # model_effNet.train()

        total_batch_loss = 0.0
        get_corrects = 0.0

        for i, data in enumerate(train_loader):
            # if i > 3:
            #     break
            # start = time.time()
            with torch.autograd.set_detect_anomaly(True):
                optimizer_model.zero_grad()
                # optimizer_model_effNet.zero_grad()

            # Prepare sample and target
            img1, img2, label_image_1, label_image_2 = data  # label_0 is the label of image_0 and label_1 is for image_1
            img1, img2, label_image_1, label_image_2 = img1.to(device), img2.to(device), label_image_1.to(device), label_image_2.to(device)

            
            label_pair = label_image_1[:] == label_image_2[:]  # label of pairs: 1 if the two images in the pair are
            # of the same class, 0 if the images belong to two different classes
            label_pair = label_pair.long()

            class_op_1 = mobilenet(img1)
            class_op_2 = mobilenet(img2)
            softmax = torch.nn.Softmax(dim=1)
            # prob1 = softmax(class_op_1)
            prob2 = softmax(class_op_2)

            # latent_1 = model_effNet(img1)
            # latent_2 = model_effNet(img2)
            # contrastive_loss = criterion_contrastive(latent_1, latent_2, label_pair)

            # class_op_1, y1o_softmax = model(latent_1)
            # class_op_2, y2o_softmax = model(latent_2)

            loss_cross_entropy_1 = loss_cross(class_op_1, label_image_1)
            loss_cross_entropy_2 = loss_cross(class_op_2, label_image_2)
         
            # get_corrects += torch.sum(torch.logical_and(torch.max(y1o_softmax, 1)[1] == label_image_1, torch.max(y2o_softmax, 1)[1] == label_image_2))
            correct_preds_1 = (class_op_1.argmax(1) == label_image_1).type(torch.float)
            correct_preds_2 = (class_op_2.argmax(1) == label_image_2).type(torch.float)
            get_corrects += torch.sum(torch.logical_and(correct_preds_2, correct_preds_1))

            loss_total = (torch.mean(prob2, dim=0)[0] * loss_cross_entropy_1 +
                          torch.mean(prob2, dim=0)[1] * loss_cross_entropy_2)  # + contrastive_loss
            # loss_total = loss_cross_entropy_1 + loss_cross_entropy_2  # + contrastive_loss
            # print('loss_total-----------------------------', loss_total)

            loss_total.backward()
            optimizer_model.step()
            # optimizer_model_effNet.step()

            # Update step
            total_batch_loss = loss_total
            losses.update(total_batch_loss.data.item(), img1.size(0))

            torch.cuda.empty_cache()

            # print("took ", time.time() - start, 'seconds')
        
        variable_acc = get_corrects.item() / dataset_sizes 

        # Log progress; print after every epochs into the console
        print('Epoch: [{:.4f}] \t The loss of this epoch is: {:.4f} \t The accuracy of this epoch is: {:.4f} '.format(epoch, losses.avg, variable_acc))

        if not os.path.isdir('./models'):
            os.makedirs('./models')
        if variable_acc > best_acc:
            print("Here the training accuracy got reduced, hence printing")
            print('Current best epoch accuracy is {:.4f}'.format(variable_acc), 'previous best was {}'.format(best_acc))
            best_acc = variable_acc
            torch.save({
                # 'model_state_dict': model.state_dict(),
                # 'model3D_state_dict': model_effNet.state_dict()},
                'model_state_dict': mobilenet.state_dict(),
                'model3D_state_dict': mobilenet.state_dict()},
                'models/mobilenet_v4.ckpt')

        # save the losses avg in .csv file
        if not os.path.isdir('./loss_1'):
            os.makedirs('./loss_1')
        with open('./loss_1/' + "loss_mobilenet_v4.csv", 'a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, losses.avg, variable_acc])
           
        my_lr_scheduler_model.step()
        # my_lr_scheduler_model_effNet.step()


if __name__ == '__main__':
    main()
