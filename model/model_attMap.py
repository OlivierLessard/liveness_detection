import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


# the only difference of this code with the one in "model.py" is that here the final ouput has 3 channels


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB(self.convB(self.convA(torch.cat([up_x, concat_with], dim=1))))


class Decoder(nn.Module):
    def __init__(self, num_features=1664, decoder_width=1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(skip_input=features // 1 + 256, output_features=features // 2)
        self.up2 = UpSample(skip_input=features // 2 + 128, output_features=features // 4)
        self.up3 = UpSample(skip_input=features // 4 + 64, output_features=features // 8)
        self.up4 = UpSample(skip_input=features // 8 + 64, output_features=features // 16)
        # self.up4 = UpSample(skip_input=features // 8 + 3, output_features=features // 16)

        # self.conv3 = nn.Conv2d(features // 16, 3, kernel_size=3, stride=1, padding=1)  # here, I have changed the
        # # number of channels from 1 to 3
        self.conv3 = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block_int, x_block0, x_block1, x_block2, x_block3, x_block4 = features[0], features[3], features[4], features[6], features[8], features[
            12]

        x_d0 = self.conv2(F.relu(x_block4))
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        return self.conv3(x_d4)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.original_model = models.densenet169(pretrained=False)

    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
            # print('v', v)
            features.append(v(features[-1]))
        return features


class SeparableConv2d(nn.Module):
    def __init__(self, c_in, c_out, ks, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.c = nn.Conv2d(c_in, c_in, ks, stride, padding, dilation, groups=c_in, bias=bias)
        self.pointwise = nn.Conv2d(c_in, c_out, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.c(x)
        x = self.pointwise(x)
        return x


class RegressionMap(nn.Module):
    def __init__(self, c_in):
        super(RegressionMap, self).__init__()
        self.c = SeparableConv2d(c_in, 1, 3, stride=1, padding=1, bias=False)
        self.s = nn.Sigmoid()

    def forward(self, x):
        mask = self.c(x)
        mask = self.s(mask)
        return mask


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        
        self.map = RegressionMap(1664) #  1280


    def forward(self, input):
        input = input.repeat(1, 3, 1, 1)
        encoder_op_1 = self.encoder(input)
        x_latent = encoder_op_1[-1] # encoder_op_1[-4]
        mask = self.map(x_latent) # layer 9
        x_mask = x_latent * mask
        return x_latent, x_mask, mask


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Fully connected layer
        self.fc1 = nn.Linear(1664*7*7, 1024)   # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
        self.fc2 = nn.Linear(1024, 128)       # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = nn.Linear(128, 32)        # convert matrix with 84 features to a matrix of 10 features (columns)
        self.fc4 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = x.view(-1, 1664*7*7) # 1280*15*15
        # FC-1, then perform ReLU non-linearity
        x = nn.functional.relu(self.fc1(x))
        # x = nn.BatchNorm2d(x)
        # FC-2, then perform ReLU non-linearity
        x = nn.functional.relu(self.fc2(x))
        # FC-3 then perform ReLU non-linearity
        x = nn.functional.relu(self.fc3(x))
        # FC-4
        x = self.fc4(x)
        y1o_softmax = F.softmax(x, dim=1)
        return x, y1o_softmax
