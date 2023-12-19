from torch import nn
from .base_model import CriticsBaseModel
import torch
from utils.utils import add_sn
from .PairSampleModule import PairSampleModule


class LatePairCriticsModel(CriticsBaseModel):
    def __init__(self,opts):
        super().__init__(opts)
        '''
        Model is modified from DCGAN-for-WGANGP (code-base:https://github.com/martinarjovsky/WassersteinGAN/blob/master/models/dcgan.py)
        Use the critics formulation of Geometric-GAN 
        Critics=Activation(<w,f(x)>)
        CriticsBaseModel corresponds to f
        f(x)=h([g(x1),g(x2)])
        [x1,x2] is a components pair sampled independently or dependently
        h is chosen to be MLP
        g is chosen to be DCGAN_D
        opts:
          critics_model_opts:
            input_f_dim
            input_t_dim
            ndf
            n_extra_layers
            memory_bank
            bank_size
        '''
        self.opts=opts
        self.pair_sample_module = PairSampleModule(opts)
        nc = 1
        ndf = opts.ndf
        n_extra_layers = opts.n_extra_layers
        assert opts.input_f_dim == opts.input_t_dim and opts.input_f_dim % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial:{0}-{1}:conv'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial:{0}:leakyrelu'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = opts.input_f_dim // 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}:{1}:conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            if opts.normalization == "layernorm":
                main.add_module('extra-layers-{0}:{1}:layernorm'.format(t, cndf),
                                nn.LayerNorm([cndf, csize, csize]))
            elif opts.normalization == "batchnorm":
                main.add_module('extra-layers-{0}:{1}:batchnorm'.format(t, cndf),
                                nn.BatchNorm2d(cndf))
            elif opts.normalization == "spectralnorm":
                pass
            else:
                raise AttributeError("Unkown Normalization Layer Type")
            main.add_module('extra-layers-{0}:{1}:relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))
        layer = 1
        while csize > 4:
            in_feat = cndf
            out_feat = min(cndf * 2,1024)
            main.add_module('pyramid{2}:{0}-{1}:conv'.format(in_feat, out_feat,layer),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            csize = csize // 2
            if opts.normalization == "layernorm":
                main.add_module('pyramid{1}:{0}:layernorm'.format(out_feat, layer),
                                nn.LayerNorm([out_feat, csize, csize]))
            elif opts.normalization == "batchnorm":
                main.add_module('pyramid{1}:{0}:batchnorm'.format(out_feat, layer),
                                nn.BatchNorm2d(out_feat))
            elif opts.normalization=="spectralnorm":
                pass
            else:
                raise AttributeError("Unkown Normalization Layer Type")

            main.add_module('pyramid{1}:{0}:leakyrelu'.format(out_feat,layer),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = min(cndf * 2,1024)
            layer +=1

        # state size. K x 4 x 4
        assert csize == 4

        main.add_module('pooling',
                        nn.AdaptiveAvgPool2d((1,1)))

        main.add_module('flatten',
                        nn.Flatten(start_dim=1, end_dim=3))  # (N,K*1)
        self.g = main

        if opts.normalization == "batchnorm":
            self.h = nn.Sequential(nn.Linear(cndf*2 , cndf ),
                                   nn.BatchNorm1d(cndf ),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(cndf , cndf ),
                                   nn.BatchNorm1d(cndf ),
                                   nn.LeakyReLU(0.2, inplace=True))
        elif opts.normalization == "spectralnorm":
            self.h = nn.Sequential(nn.Linear(cndf*2 , cndf ),
                                   nn.LeakyReLU(0.2, inplace=True),
                                   nn.Linear(cndf , cndf ),
                                   nn.LeakyReLU(0.2, inplace=True))
        else:
            raise AttributeError("Unkown Normalization Layer Type")



        self.w = nn.Linear(cndf , 1, bias=False)
        self.b = nn.Parameter(torch.zeros(1))

        if opts.normalization == "spectralnorm":
            print("Critics is using SpectralNorm")
            add_sn(self.g)
            add_sn(self.h)
        elif opts.normalization == "layernorm":
            print("Critics is using LayerNorm")
        elif opts.normalization == "batchnorm":
            print("Critics is using BatchNorm")

    def forward(self,inputs):
        '''
        -------Inputs------
        -SepResults
        est_components:B,NumMix,Objects*C,AudLen
        est_score_map,est_mel_mask,est_mel_mag:B,NumMix,Objects*C,F,T
        valid_nums:B,NumMix,Objects*C
        -------------------

        -------Outputs------
        independent_pair:(N,2,F,T)
        dependent_pair:(M,2,F,T)
        independent_pair_feature=f(x):N,D
        dependent_pair_feature=f(x): M,D
        independent_pair_score=<w,f(x)>:N
        dependent_pair_score=<w,f(x)>: M
        -------------------
        '''

        outputs={}
        independent_pair,dependent_pair=self.pair_sample_module(inputs)

        outputs["independent_pair"] = independent_pair
        outputs["dependent_pair"] = dependent_pair

        outputs["independent_pair_feature"],outputs["independent_pair_score"]=self.pair_forwards(independent_pair)
        outputs["dependent_pair_feature"],outputs["dependent_pair_score"]=self.pair_forwards(dependent_pair)
        return outputs


    def pair_forwards(self,mel_mag_pairs):
        #mel_mag_pairs:(N,2,F,T)
        N=mel_mag_pairs.size(0)
        mel_mag_pairs=mel_mag_pairs.flatten(end_dim=1).unsqueeze(1)
        component_feature=self.g(mel_mag_pairs)#(N*2,D1)

        pair_feature=self.h(component_feature.view(N,-1))#(N,D_pair)

        pair_score=self.w(pair_feature)+self.b

        return pair_feature,pair_score

