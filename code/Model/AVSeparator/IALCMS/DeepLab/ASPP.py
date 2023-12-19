import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import get_normalization_for_G
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,normalization="batchnorm"):
        super(_ASPPModule, self).__init__()
        layers=[]
        layers.extend([nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False),
                       *get_normalization_for_G(planes, normalization),
                       nn.LeakyReLU()])

        self.model=nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ASPP(nn.Module):
    def __init__(self, inplanes, dilations,out_dim):
        super(ASPP, self).__init__()

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, out_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if not m.weight is None:
                    m.weight.data.fill_(1)
                else:
                    print("ASPP has not weight: ", m)

                if not m.bias is None:
                    m.bias.data.zero_()
                else:
                    print("ASPP has not bias: ", m)


def build_aspp(backbone, output_stride):
    return ASPP(backbone, output_stride)
