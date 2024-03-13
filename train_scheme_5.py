import gc
from operator import ne
import matplotlib.pyplot as plt
import statistics
import time
import argparse
import datetime
import os
import numpy as np
import torch
import sys
import csv
import torch.nn as nn
import torchvision.utils as vutils
from torchvision import transforms
import torchvision.datasets as dataset
from torch.utils.data import DataLoader, Dataset
from model.gan_model import Discriminator, Generator
from model.model_attMap import Decoder
from loss import ContrastiveLoss, ssim, AdversarialLoss
from dataloaders.data_5 import SiameseNetworkDataset, CsvDataset, CsvDataset_half
from torch.optim import lr_scheduler


def get_pair_training_dataloader(batch_size):
    train_csv_dir = './face_db/training' + '.csv'
    # transform1 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    transform1 = transforms.Compose([transforms.Resize((240, 240)), transforms.Grayscale(), transforms.ToTensor()])
    transform2 = transforms.Compose([transforms.Resize((120, 120)), transforms.Grayscale(), transforms.ToTensor()])
    training_dataset = CsvDataset_half(csv_file=train_csv_dir, transform1=transform1, transform2=transform2, should_invert=False)
    train_loader = DataLoader(training_dataset, shuffle=True, num_workers=0,
                              batch_size=batch_size, pin_memory=True)  # num_workers=4 * torch.cuda.device_count()
    dataset_sizes = len(train_loader.dataset)
    return dataset_sizes, train_loader


def get_pair_validation_dataloader(batch_size):
    train_csv_dir = './face_db/validation' + '.csv'
    # transform1 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    transform1 = transforms.Compose([transforms.Resize((240, 240)), transforms.Grayscale(), transforms.ToTensor()])
    training_dataset = CsvDataset(csv_file=train_csv_dir, transform1=transform1, should_invert=False)
    train_loader = DataLoader(training_dataset, shuffle=True, num_workers=0,
                              batch_size=batch_size, pin_memory=True)  # num_workers=4 * torch.cuda.device_count()
    dataset_sizes = len(train_loader.dataset)
    return dataset_sizes, train_loader


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Fraud Detection in Identity Card')
    parser.add_argument('--root', type=int, help='set the root of dataset')
    parser.add_argument('--epochs', default=60, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float, help='initial learning rate')  # default=0.00001
    parser.add_argument('--bs', default=1, type=int, help='batch size')  # 3 gives OOM
    parser.add_argument('--beta1', default=0.5, type=float, help='hyperparam for Adam optimizers')

    args = parser.parse_args()

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")
    ngpu = torch.cuda.device_count()

    # Create the Discriminator
    netD = Discriminator().to(device)
    print('Discriminator Model created.')
    if (device.type == 'cuda') and (ngpu > 1):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netD = nn.DataParallel(netD.to(device))

    # Create the Generator
    netG = Generator().to(device)
    print('Generator Model created.')
    if (device.type == 'cuda') and (ngpu > 1):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netG = nn.DataParallel(netG.to(device))

    # Create the Decoder
    decoder = Decoder().to(device)
    print('Decoder Model created.')
    if (device.type == 'cuda') and (ngpu > 1):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        decoder = nn.DataParallel(netG.to(device))

    print('model and cuda mixing done')

    # Training parameters
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    D_lr_scheduler = lr_scheduler.StepLR(optimizerD, step_size=20, gamma=0.1)
    G_lr_scheduler = lr_scheduler.StepLR(optimizerG, step_size=20, gamma=0.1)
    decoder_lr_scheduler = lr_scheduler.StepLR(optimizer_decoder, step_size=20, gamma=0.1)

    best_loss = 100.0
    best_acc = 0.0
    batch_size = args.bs

    # Load data
    train_dataset_sizes, train_loader = get_pair_training_dataloader(batch_size)
    val_dataset_sizes, val_loader = get_pair_validation_dataloader(batch_size)
    print(train_dataset_sizes)
    print("Total number of batches in train loader are :", len(train_loader))

    # Loss
    criterion_contrastive = ContrastiveLoss()
    loss_cross = torch.nn.CrossEntropyLoss()
    l1_criterion = nn.L1Loss()
    d_label_real_img = torch.LongTensor([1]*batch_size).to(device)
    d_label_fake_img = torch.LongTensor([0]*batch_size).to(device)

    # Start training...
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        G_losses, D_losses = [], []
        train_corrects, val_corrects = 0.0, 0.0

        netD.train()
        netG.train()
        decoder.train()
        for i, data in enumerate(train_loader):
            # Prepare sample and target
            img1, img1_half, img2, img2_half, label_1, label_2 = data
            img1, img1_half, img2, img2_half, label_1, label_2 = (img1.to(device), img1_half.to(device), img2.to(device),
                                                                  img2_half.to(device), label_1.to(device), label_2.to(device))
            
            latent_1, x_mask_1, mask_1, encoder_op_1 = netG(img1)
            latent_2, x_mask_2, mask_2, encoder_op_2 = netG(img2)
            decoder_op_1 = decoder(encoder_op_1)
            decoder_op_2 = decoder(encoder_op_2)

            ############################
            # Calculate the contrastive loss
            ############################
            label_pair = label_1[:] == label_2[:]
            label_pair = label_pair.long()
            contrastive_loss_densNet = criterion_contrastive(latent_1, latent_2, label_pair)

            ############################
            # reconstruction loss
            ############################
            loss_image_direct_1 = l1_criterion(decoder_op_1, img1_half)
            loss_ssim_image_direct_1 = torch.clamp(
                (1 - ssim(decoder_op_1.float(), img1_half.float(), val_range=1)) * 0.5, 0, 1)
            loss_image_total_1 = (1.0 * loss_image_direct_1) + (0.1 * loss_ssim_image_direct_1)
            loss_image_direct_2 = l1_criterion(decoder_op_2, img2_half)
            loss_ssim_image_direct_2 = torch.clamp(
                (1 - ssim(decoder_op_2.float(), img2_half.float(), val_range=1)) * 0.5, 0, 1)
            # clamps all the input elements into the range [ min, max ]
            loss_image_total_2 = (1.0 * loss_image_direct_2) + (0.1 * loss_ssim_image_direct_2)
            reconstruction_loss = loss_image_total_1 + loss_image_total_2
            # print('loss_image_total_2-----------------------------', loss_image_total_2)

            ############################
            # (1) Update D network: maximize log(D(x)))     # discriminator adversarial loss
            ###########################
            # Calculate loss on all-real batch
            optimizerD.zero_grad()
            real_vid_feat, y1o_softmax = netD(x_mask_1)
            dis_real_loss = loss_cross(real_vid_feat, d_label_real_img)
            dis_real_loss.backward(retain_graph=True)

            # Calculate loss on all-fake batch
            fake_vid_feat, y2o_softmax = netD(x_mask_2.detach())
            dis_fake_loss = loss_cross(fake_vid_feat, d_label_fake_img)
            dis_fake_loss.backward(retain_graph=True)

            D_losses.append(dis_fake_loss.item())

            # Update D
            optimizerD.step()
                        
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            # generator adversarial loss
            ###########################
            optimizerG.zero_grad()
           
            gen_fake_feat, y2o_softmax_fake = netD(x_mask_2)
            gen_real_feat, y1o_softmax_real = netD(x_mask_1)
            gen_fake_loss = loss_cross(gen_fake_feat, d_label_real_img)
            gen_real_loss = loss_cross(gen_real_feat, d_label_fake_img)
            gen_loss = (gen_fake_loss + gen_real_loss + contrastive_loss_densNet + reconstruction_loss)/4
            G_losses.append(gen_loss.item())

            # Update G
            gen_loss.backward()
            optimizerG.step()
            optimizer_decoder.step()  # Update decoder TODO

            train_corrects += torch.sum(torch.logical_and(torch.max(y1o_softmax, 1)[1] == label_1, torch.max(y2o_softmax, 1)[1] == label_2))

            # print('loss_total---',  dis_loss, gen_loss)

        netD.eval()
        netG.eval()
        for i, data in enumerate(val_loader):

            # Prepare sample and target
            img1, img2, label_1, label_2 = data
            img1, img2, label_1, label_2 = img1.to(device), img2.to(device), label_1.to(device), label_2.to(device)

            # inference
            _, x_mask_1, mask_1, encoder_op_1 = netG(img1)
            _, x_mask_2, mask_2, encoder_op_2 = netG(img2)
            real_vid_feat, y1o_softmax = netD(x_mask_1)
            fake_vid_feat, y2o_softmax = netD(x_mask_2.detach())

            val_corrects += torch.max(y1o_softmax, 1)[1] == label_1
            val_corrects += torch.max(y2o_softmax, 1)[1] == label_2

        train_accuracy = train_corrects.item() / train_dataset_sizes
        val_accuracy = val_corrects.item() / val_dataset_sizes
        print('Epoch: [{:.4f}] \t The dis_loss of this epoch is: {:.4f} \t The gen_loss of this epoch is: {:.4f}\t Train accuracy is: {:.4f} Val accuracy is: {:.4f} '.format(epoch, statistics.mean(D_losses), statistics.mean(G_losses), train_accuracy, val_accuracy))

        if not os.path.isdir('./trained_models_for_paper'):
            os.makedirs('./trained_models_for_paper')
        if val_accuracy > best_acc:
            print("Here the training accuracy got reduced, hence printing")
            print('Current best epoch val accuracy is {:.4f}'.format(val_accuracy), 'previous best was {}'.format(best_acc))
            best_acc = val_accuracy
            dir = 'trained_models_for_paper'
            if not os.path.isdir(dir):
                os.makedirs(dir)
            torch.save({
                'model_G_state_dict': netG.state_dict(),
                'model_D_state_dict': netD.state_dict()},
                dir + '/scheme5_net_G_D_scheme_5.ckpt')

        # save the losses avg in .csv file
        if not os.path.isdir('loss'):
            os.makedirs('loss')
        with open('./loss/' + "scheme5_loss_G_D_scheme_5.csv", 'a') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, statistics.mean(D_losses),  statistics.mean(G_losses), train_accuracy, val_accuracy])

        D_lr_scheduler.step()
        G_lr_scheduler.step()


if __name__ == '__main__':
    main()
