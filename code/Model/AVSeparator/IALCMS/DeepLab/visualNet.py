import torch
from functools import partial
from .deeplabv3plus import Deeplabv3plus
from torch import nn
from torch.nn import functional


class VisualNet(nn.Module):
    def __init__(self,args):
        super(VisualNet, self).__init__()
        #########Base Net #########
        dargs=args.deeplab
        self.basenet=Deeplabv3plus(decoder_conv_num=dargs.decoder.num,
                              out_dim=dargs.decoder.out_dim,
                              shallow_layer=dargs.shallow.layer,
                              reduced_dim=dargs.shallow.reduced_dim,
                              aspp_args=dargs.aspp,
                              freeze_backbone=dargs.freeze
                              )

        self.dropout=args.dropout
        #########Common Space MLP #########
        in_dim = dargs.decoder.out_dim
        cargs=args.common_space_mlp
        blocks = []
        for i in range(cargs.hidden_layer):
            blocks.extend([nn.Conv2d(in_dim,in_dim,1),
                           nn.ReLU(),
                           nn.BatchNorm2d(in_dim)])
        blocks.extend([nn.Conv2d(in_dim, cargs.out_dim, 1)])
        self.common_space_mlp=nn.Sequential(*blocks)


    def forward(self,img):
        f,_=self.basenet(img)
        common_f=self.common_space_mlp(f)
        common_f=functional.normalize(common_f,p=2,dim=1)
        dropout_f=functional.dropout(f,p=self.dropout)
        dropout_common_f=self.common_space_mlp(dropout_f)
        dropout_common_f =functional.normalize(dropout_common_f,p=2,dim=1)
        return common_f,dropout_common_f
