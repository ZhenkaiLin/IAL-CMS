import torch
from torch import nn
import numpy as np

import torch.nn.functional as F

from .ASPP import ASPP
from .resnet18 import ResNet18,ResNet18Pretrained
class Deeplabv3plus(nn.Module):
    def __init__(self,decoder_conv_num,out_dim,shallow_layer,reduced_dim,aspp_args,freeze_backbone):
        super(Deeplabv3plus, self).__init__()
        self.backbone=ResNet18Pretrained()
        if freeze_backbone:
            self.backbone.requires_grad_(False)
        self.aspp=ASPP(512,**aspp_args)

        shallow_dims=[64,64,128,256,512]
        shallow_dim=shallow_dims[shallow_layer-1]
        self.reduce_block=nn.Sequential(
            nn.Conv2d(shallow_dim,reduced_dim,1),
            nn.ReLU(reduced_dim),
            nn.BatchNorm2d(reduced_dim)
        )
        dec_layers=[nn.Conv2d(reduced_dim+aspp_args.out_dim,out_dim,3,padding=1),
            nn.ReLU(out_dim),
            nn.BatchNorm2d(out_dim)]
        for i in range(decoder_conv_num-1):
            dec_layers.extend([
                nn.Conv2d(out_dim,out_dim,3,padding=1),
                nn.ReLU(out_dim),
                nn.BatchNorm2d(out_dim)])
        self.decoder=nn.Sequential(*dec_layers)

        self.shallow_layer=shallow_layer

    def forward(self, x):
        out=self.backbone.forward_as_dict(x)
        shallow=out["conv"+str(self.shallow_layer)]
        deep=out["conv5"]

        f1=self.aspp(deep)
        f1=F.interpolate(f1, size=shallow.size()[2:], mode='bilinear', align_corners=True)

        f2=self.reduce_block(shallow)

        f=torch.cat([f1,f2],dim=1)

        f=self.decoder(f)

        return f,deep

