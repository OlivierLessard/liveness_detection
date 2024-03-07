import torch
import torch.nn as nn
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


def conv_1x1_bn(inp, oup, stride, relu=True):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential()
    )


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class SqueezeExcitation(nn.Module):
    def __init__(self, inplanes, se_planes):
        super(SqueezeExcitation, self).__init__()
        self.reduce_expand = nn.Sequential(
            nn.Conv2d(inplanes, se_planes, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_planes, inplanes, 1),
            h_sigmoid()
        )

    def forward(self, x):
        x_se = F.adaptive_avg_pool2d(x, 1)
        x_se = self.reduce_expand(x_se)
        return x * x_se


class ResidualUnit(nn.Module):
    def __init__(self, inp, oup, stride, use_se):
        super(ResidualUnit, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_se = use_se
        self.identity = nn.Sequential()
        if stride == 1 and inp != oup:
            self.identity = nn.Conv2d(inp, oup, 1, 1, bias=False)
            self.identity_bn = nn.BatchNorm2d(oup)

        self.conv1 = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(oup)

        if use_se:
            self.se = SqueezeExcitation(oup, inp)

    def forward(self, x):
        if hasattr(self, 'identity_bn'):
            identity = self.identity_bn(self.identity(x))
        else:
            identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.use_se:
            x = self.se(x)
        return self.relu(x + identity)


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=2, multiplier=1.0):
        super(MobileNetV3Small, self).__init__()
        self.cfgs = [
            # k, exp, c,  se,     nl,  s,
            [3, 16,  16,  True, 'RE', 1],
            [3, 72,  24,  False, 'RE', 1],
            [3, 88,  24,  False, 'RE', 1],
            [5, 96,  40,  True,  'HS', 1],
            [5, 240, 40,  True,  'HS', 1],
            [5, 240, 40,  True,  'HS', 1],
            [5, 120, 48,  True,  'HS', 1],
            [5, 144, 48,  True,  'HS', 1],
            [5, 288, 96,  True,  'HS', 1],
            [5, 576, 96,  True,  'HS', 1],
            [5, 576, 96,  True,  'HS', 1],
        ]

        input_channel = int(16 * multiplier) if multiplier > 1.0 else 16
        self.last_channel = int(1024 * multiplier) if multiplier > 1.0 else 1024

        self.features = [conv_3x3_bn(3, input_channel, 2)]  # Initial conv layer with stride 2
        # Building inverted residual blocks
        for k, exp, c, se, nl, s in self.cfgs:
            output_channel = int(c * multiplier)
            exp_channel = int(exp * multiplier)
            self.features.append(ResidualUnit(input_channel, output_channel, s, use_se=se))
            input_channel = output_channel

        # Building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel, 1))
        self.features = nn.Sequential(*self.features)
        # Building classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.last_channel, 1280),
            h_swish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

        self._initialize_weights()

        self.convolution_map_layer = nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=(3, 3), stride=(1, 1),
                                               padding=(1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)  # [B, 1024, 112, 112]

        # compute attention weights
        map = self.convolution_map_layer(x)  # [B, 1, 112, 112]
        attention_map = self.sigmoid(map)

        # multiply with the attention weights
        x = x * attention_map.repeat(1, 1024, 1, 1)  # [B, 1024, 112, 112]

        # classify
        scores = x.mean([2, 3])  # [B, 1024]
        return scores

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
