import torch
from torch import nn
from torch.nn import functional
from functools import partial

class ParamSigmoid(nn.Module):
    def __init__(self):
        super(ParamSigmoid, self).__init__()
        self.w=nn.Parameter(torch.tensor(1.))
        self.b=nn.Parameter(torch.tensor(-0.2))

    def forward(self,x):
        return functional.sigmoid(x*self.w+self.b)
