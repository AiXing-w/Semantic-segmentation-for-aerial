import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

# U-Net
class up_sample(nn.Module):
    def __init__(self, channel1, channel2):
        super(up_sample, self).__init__()
        self.up = nn.ConvTranspose2d(channel1, channel1, kernel_size=2, stride=2, bias=False)
        self.conv1 = nn.Conv2d(channel1 + channel2, channel2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel2, channel2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input1, input2):
        if input1.shape[-2:] != input2.shape[-2:]:
            input1 = self.up(input1)

        image_size = input2.shape[-2:]
        input1 = F.interpolate(input1, image_size)
        outputs = torch.cat([input1, input2], dim=1)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class U_net(nn.Module):
    def __init__(self, backbone='VGG16', channels=[64, 128, 256, 512, 512], out_channel=6):
        super(U_net, self).__init__()
        if backbone == 'VGG16':
            vgg16 = models.vgg16(pretrained=True)
            backbone = list(vgg16.children())[0]
            self.b1 = nn.Sequential(*list(backbone.children())[:5])
            self.b2 = nn.Sequential(*list(backbone.children())[5:10])
            self.b3 = nn.Sequential(*list(backbone.children())[10:17])
            self.b4 = nn.Sequential(*list(backbone.children())[17:24])
            self.b5 = nn.Sequential(*list(backbone.children())[24:])

        self.up_sample2 = up_sample(channels[1], channels[0])
        self.up_sample3 = up_sample(channels[2], channels[1])
        self.up_sample4 = up_sample(channels[3], channels[2])
        self.up_sample5 = up_sample(channels[4], channels[3])

        self.uplayer = nn.ConvTranspose2d(channels[0], out_channel, kernel_size=2, stride=2, bias=False)

    def forward(self, X):
        X1 = self.b1(X)
        X2 = self.b2(X1)
        X3 = self.b3(X2)
        X4 = self.b4(X3)
        X5 = self.b5(X4)

        output = self.up_sample5(X5, X4)
        output = self.up_sample4(output, X3)
        output = self.up_sample3(output, X2)
        output = self.up_sample2(output, X1)
        output = self.uplayer(output)
        return output

# deeplabv3+


class Residual(nn.Module):
    # 残差
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)

        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)

        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    # 残差块
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))

    return blk


def resNet18(in_channels):
    # resnet18
    b1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                       )

    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))

    b3 = nn.Sequential(*resnet_block(64, 128, 2))

    b4 = nn.Sequential(*resnet_block(128, 256, 2))

    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    net = nn.Sequential(b1, b2, b3, b4, b5)

    return net


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            #                 nn.AdaptiveAvgPool2d((1, 1)),  # (b, c, r, c)->(b, c, 1, 1)
            nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True),  # (b, c_out, 1, 1)
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()

        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class deeplabv3(nn.Module):
    def __init__(self, in_channels, num_classes, bonenet='resNet18'):
        super(deeplabv3, self).__init__()
        if bonenet == 'resNet18':
            self.bonenet = 'resNet18'
            self.layers = resNet18(3)
            low_level_channels = 64
            high_level_channels = 512
            low_out_channels = 64
            high_out_channels = 256

        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, low_out_channels, 1),
            nn.BatchNorm2d(low_out_channels),
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(high_level_channels, high_out_channels)
        self.cat_conv = nn.Sequential(
            nn.Conv2d(256 + 64, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)
        if self.bonenet == 'resNet18':
            self.up_sample = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, padding=2, stride=2,
                                                bias=False)

    def forward(self, X):
        img_size = X.shape[-2:]
        if self.bonenet == 'resNet18':
            X = self.layers[:2](X)

        short_cut = self.shortcut_conv(X)
        if self.bonenet == 'resNet18':
            X = self.layers[2:](X)

        aspp = self.aspp(X)
        aspp = F.interpolate(aspp, size=(short_cut.shape[-2], short_cut.shape[-1]), mode='bilinear', align_corners=True)

        concat = self.cat_conv(torch.cat([aspp, short_cut], dim=1))
        ans = self.cls_conv(concat)
        ans = self.up_sample(ans)
        #         ans = F.interpolate(ans, size=(img_size[0], img_size[1]), mode='bilinear', align_corners=True)
        return ans