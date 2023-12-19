import torch
from torch import nn
import numpy as np

import torch.nn.functional as F
from functools import partial

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, first_dilation=None, dilation=1):
        super(ResBlock, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        if first_dilation == None: first_dilation = dilation

        self.bn_branch2a = nn.BatchNorm2d(in_channels)

        self.conv_branch2a = nn.Conv2d(in_channels, mid_channels, 3, stride,
                                       padding=first_dilation, dilation=first_dilation, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(mid_channels)

        self.conv_branch2b1 = nn.Conv2d(mid_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)

        x_bn_relu = branch2

        if not self.same_shape:
            branch1 = self.conv_branch1(branch2)
        else:
            branch1 = x

        branch2 = self.conv_branch2a(branch2)
        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.conv_branch2b1(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)

class ResBlock_bot(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.):
        super(ResBlock_bot, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        self.bn_branch2a = nn.BatchNorm2d(in_channels)
        self.conv_branch2a = nn.Conv2d(in_channels, out_channels//4, 1, stride, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(out_channels//4)
        self.dropout_2b1 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b1 = nn.Conv2d(out_channels//4, out_channels//2, 3, padding=dilation, dilation=dilation, bias=False)

        self.bn_branch2b2 = nn.BatchNorm2d(out_channels//2)
        self.dropout_2b2 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b2 = nn.Conv2d(out_channels//2, out_channels, 1, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)
        x_bn_relu = branch2

        branch1 = self.conv_branch1(branch2)

        branch2 = self.conv_branch2a(branch2)

        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b1(branch2)
        branch2 = self.conv_branch2b1(branch2)

        branch2 = self.bn_branch2b2(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b2(branch2)
        branch2 = self.conv_branch2b2(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)

class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()


        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 64, 64, stride=2)
        self.b2_1 = ResBlock(64, 64, 64)

        self.b3 = ResBlock(64, 128, 128, stride=2)
        self.b3_1 = ResBlock(128, 128, 128)

        self.b4 = ResBlock(128, 256, 256, stride=2)
        self.b4_1 = ResBlock(256, 256, 256)

        self.b5 = ResBlock(256, 512, 512, stride=1, dilation=2)
        self.b5_1 = ResBlock(512, 512, 512, dilation=2)

        self.bn5=nn.BatchNorm2d(512)
        # OS:8 4

    def forward(self, x):
        return self.forward_as_dict(x)['conv5']

    def forward_as_dict(self, x):

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        conv2=x

        x = self.b3(x)
        x = self.b3_1(x)
        conv3 = x

        x = self.b4(x)
        x = self.b4_1(x)
        conv4=x

        x = self.b5(x)
        x = self.b5_1(x)

        conv5 = F.relu(self.bn5(x))

        return dict({'conv2': conv2,'conv3': conv3,'conv4': conv4, 'conv5': conv5})

import torchvision
class ResNet18Pretrained(nn.Module):
    def __init__(self,OS=16):
        super(ResNet18Pretrained, self).__init__()
        original_resnet=torchvision.models.resnet18(pretrained=True)
        childrens = list(original_resnet.children())

        self.layer1=nn.Sequential(*childrens[:3])
        self.layer2 = nn.Sequential(*childrens[3:5])
        self.layer3 = childrens[5]
        self.layer4 = childrens[6]
        self.layer5= childrens[7]

        if OS == 8:
            self.layer4.apply(
                partial(self._nostride_dilate, dilate=2))
            self.layer5.apply(
                partial(self._nostride_dilate, dilate=4))
        elif OS == 16:
            self.layer5.apply(
                partial(self._nostride_dilate, dilate=2))
        elif OS==32:
            pass
        else:
            raise AttributeError("Unknown resnet oupit stride.")


    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward_as_dict(self,x):
        # (B,AudLen)
        x=self.layer1(x)

        x = self.layer2(x)
        conv2 = x

        x = self.layer3(x)
        conv3 = x

        x = self.layer4(x)
        conv4 = x

        x = self.layer5(x)
        conv5 = x

        return dict({'conv2': conv2, 'conv3': conv3, 'conv4': conv4, 'conv5': conv5})

    def forward(self, x):
        return self.forward_as_dict(x)['conv5']

if __name__=="__main__":
    net=ResNet18Pretrained()
    net2=ResNet18Pretrained(OS=32)