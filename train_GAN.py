
import statistics
import argparse
import torch
import csv
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from model.model_attMap import Generator, Discriminator
from dataloaders.data_5 import CsvDataset
from torch.optim import lr_scheduler
from utils import AverageMeter
import torch.nn.functional as F

MODEL_DIR = 'trained_models_for_competition/'

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Fraud Detection in Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--epochs', default=15, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=2, type=int, help='batch size')
    parser.add_argument('--beta1', default=0.5, type=float, help='hyperparam for Adam optimizers')

    args = parser.parse_args()


    train_csv_dir = './face_db/training.csv'
    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")
    ngpu = torch.cuda.device_count()

    # Create the Discriminator
    netD = Discriminator().to(device)
    netG = Generator().to(device)
    print('Discriminator and Generator Models created.')
    if (device.type == 'cuda') and (ngpu > 1):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netD = nn.DataParallel(netD.to(device))
        netG = nn.DataParallel(netG.to(device))
  
    print('model and cuda mixing done')

    # Training parameters
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    D_lr_scheduler = lr_scheduler.StepLR(optimizerD, step_size=20, gamma=0.1)
    G_lr_scheduler = lr_scheduler.StepLR(optimizerG, step_size=20, gamma=0.1)

    best_loss = 100.0
    best_acc = 0.0
    batch_size = args.bs

    # Load data
    transform1 = transforms.Compose([transforms.Resize((240, 240)), transforms.ToTensor()])
    # transform2 = transforms.Compose([transforms.Resize((120, 120)), transforms.ToTensor()])
    training_dataset = CsvDataset(csv_file=train_csv_dir, transform1=transform1, should_invert=False)

    train_loader = DataLoader(training_dataset, shuffle=True, num_workers=1 * torch.cuda.device_count(),
                              batch_size=batch_size, pin_memory=True)
 
    print("Total number of batches in train loader are :", len(train_loader))
    dataset_sizes = len(train_loader.dataset)
    print('dataset_sizes', dataset_sizes)

    # Loss
    LOSS_L2 = nn.MSELoss().to(device)
    LOSS_CSE = nn.CrossEntropyLoss().to(device)

    
    real_imag_label = 1.0
    fake_imag_label = 0.0
    d_label_real_img = torch.cuda.LongTensor([1]*batch_size)
    d_label_fake_img = torch.cuda.LongTensor([0]*batch_size)
    # g_label_fake_img = torch.cuda.LongTensor([1]*batch_size)
    # g_label_real_img = torch.cuda.LongTensor([0]*batch_size)   
    
    # Start training...
    for epoch in range(args.epochs):
        # torch.cuda.empty_cache()
        losses = AverageMeter()
        N = len(train_loader)
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)
        # Switch to train mode
        # model.train()
        G_losses = []
        D_losses = []
        get_corrects = 0.0

        for i, data in enumerate(train_loader):

            # netD.train()
            # netG.train() 
            gen_loss = 0

            # with torch.enable_grad():   

            # Prepare sample and target
            img1, img2, label_image_1, label_image_2 = data  # label_0 is the label of image_0 and label_1 is for image_1
            img1, img2, label_image_1, label_image_2 = img1.to(device), img2.to(
                device), label_image_1.to(device), label_image_2.to(device)
            
            # label_size = label_image_1.size(0)
            # d_label_real_img = torch.cuda.LongTensor([1.0]*label_size) 
            # d_label_fake_img = torch.cuda.LongTensor([0.0]*label_size) 

            img1_gt_msk = F.interpolate(torch.abs(img1 - img1), (7, 7)).to(device)
            img2_gt_msk = F.interpolate(torch.abs(img1 - img2), (7, 7)).to(device)            
            
            _, x_mask1, mask1 = netG(img1)
            _, x_mask2, mask2 = netG(img2)
            
            mask_loss_l1_img1 = LOSS_L2(mask1, img1_gt_msk)
            mask_loss_l1_img2 = LOSS_L2(mask2, img2_gt_msk)

            optimizerD.zero_grad()
            optimizerG.zero_grad()
            # Calculate loss on all-real batch
            real_vid_feat, y1o_softmax  = netD(x_mask1)
            dis_real_loss = LOSS_CSE(real_vid_feat, d_label_real_img)
            pred_real = torch.max(real_vid_feat, dim=1)[1]
            # dis_real_loss.backward(retain_graph=True)


            # Calculate loss on all-fake batch
            fake_vid_feat, y2o_softmax = netD(x_mask2) 
            dis_fake_loss = LOSS_CSE(fake_vid_feat, d_label_fake_img)
            pred_fake = torch.max(fake_vid_feat, dim=1)[1]
            # dis_fake_loss.backward(retain_graph=True)
            lambda1 = lambda2 = 1.0

            total_real_img_entropy_loss = dis_real_loss + lambda1 * mask_loss_l1_img1
            total_fake_img_entropy_loss = dis_fake_loss + lambda2 * mask_loss_l1_img2
            
            total_dis_loss = (total_real_img_entropy_loss + total_fake_img_entropy_loss)/2.0
            total_dis_loss.backward()

            D_losses.append(total_dis_loss.item())
            # Update D
            optimizerD.step()
            optimizerG.step()
            
            # get_corrects += torch.sum(torch.logical_and(torch.max(y1o_softmax, 1)[1] == label_image_1, torch.max(y2o_softmax, 1)[1] == label_image_2))
            get_corrects += torch.sum(torch.logical_and(pred_real == label_image_1, pred_fake == label_image_2))

            # print('get_corrects', get_corrects)

            # print('loss_total---',  dis_loss, gen_loss)

        variable_acc = get_corrects.item() / dataset_sizes

        # print('Epoch: [{:.4f}] \t The loss of this epoch is: {:.4f} \t The accuracy of this epoch is: {:.4f} '.format(epoch, losses.avg, variable_acc))
        print('Epoch: [{:.4f}] \t The dis_loss of this epoch is: {:.4f} \t The accuracy of this epoch is: {:.4f} '.format(epoch, statistics.mean(D_losses), variable_acc))

        if variable_acc > best_acc:
            print("Here the training accuracy got reduced, hence printing")
            print('Current best epoch accuracy is {:.4f}'.format(variable_acc), 'previous best was {}'.format(best_acc))
            best_acc = variable_acc
            torch.save({
                'netD_state_dict': netD.state_dict(),
                'netG_state_dict': netG.state_dict()},
                MODEL_DIR + '/sch1_attMap.ckpt')
            

        # save the losses avg in .csv file
        # with open("loss_avg_module_3_advNet_v3_" + country_name + ".csv", 'a') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([epoch, statistics.mean(D_losses), statistics.mean(G_losses), variable_acc])

        with open("loss_competition/loss_sch1_attMap.csv", 'a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, statistics.mean(D_losses), variable_acc])
        
       
        D_lr_scheduler.step()
        G_lr_scheduler.step()



if __name__ == '__main__':
    main()