#(B,NumMix,C+1,D)
import torch
from torch import nn
from torch.nn import functional
from functools import partial
from .base_model.AdaptiveDifferentialBinarizationSegment import ParamSigmoid
from utils.utils import spacial_minmaxnormalize,organize_batch,flatten_batch
import pickle
def agg_multi(x,dim):
    assert dim==2
    D=x.size(dim)
    agg=x[:,:,0]
    for d in range(1,D):
        agg=x[:,:,d]*agg
    return agg.unsqueeze(dim)

class AVModule(nn.Module):
    def __init__(self,opts):
        super(AVModule, self).__init__()
        '''
        '''

        self.opts=opts

        in_dim = opts.a_in_dim
        sep_in_dim=in_dim*2
        self.sep_mlp = nn.Sequential(nn.Conv2d(sep_in_dim, sep_in_dim, kernel_size=1),
                                     nn.LeakyReLU(0.2, True),
                                     nn.BatchNorm2d(sep_in_dim),
                                     nn.Conv2d(sep_in_dim, in_dim,
                                               kernel_size=1),
                                     nn.LeakyReLU(0.2, True),
                                     nn.BatchNorm2d(in_dim)
                                     )
        aargs = opts.a_common_mlp
        assert opts.src_f_dim==aargs.out_dim and opts.v_in_dim==aargs.out_dim
        blocks = []
        hid_dim=aargs.hidden_dim
        for i in range(aargs.hidden_layer):
            blocks.extend([nn.Conv2d(in_dim,hid_dim,1),
                           nn.LeakyReLU(0.2),
                           nn.BatchNorm2d(hid_dim)])
            in_dim=hid_dim
        blocks.extend([nn.Conv2d(in_dim, aargs.out_dim,1)])
        self.a_common_mlp=nn.Sequential(*blocks)

        assert in_dim==aargs.out_dim
        self.weight_mlp = nn.Sequential(*[nn.Conv1d(aargs.out_dim, aargs.out_dim, 1),
                                          nn.LeakyReLU(0.2),
                                          nn.BatchNorm1d(aargs.out_dim),
                                          nn.Conv1d(aargs.out_dim, aargs.out_dim, 1),
                                          nn.Sigmoid()
                                          ])

        if opts.sigmoid=="params":
            self.sigmoid=ParamSigmoid()

    def weight_pooling(self,C1weight_map,fv_common):
        #weight_map:(N,H,W) fv_common(N,D,H,W) ->(N,D)
        C1weight_map1=C1weight_map.unsqueeze(1)
        C1_sp_att_fv_com=(C1weight_map1*fv_common).sum(dim=[-1,-2])#/(C1weight_map1.sum(dim=[-1,-2])+1e-8)#(N,D)
        return functional.normalize(C1_sp_att_fv_com,dim=-1,p=1)

    def forward(self,inputs):
        '''
        ------- Input ------
        fa_mix:(B,D,F,T)
        multi_scale_fa:(B,NumMix*Objects,D)
        valid_nums:(B,NumMix*Objects)
        fv_common_valid:(N,D,H,W)
        --------------------

        ------- Output ------
        -extrated_features
        visual_related_srcs_fa_com:(B,NumMix*Obejcts,D)
        sp_att_fv_com:(B,NumMix*Obejcts,D)
        visual_related_srcs_att_fa:(B,NumMix*Obejcts,D,F,T)
        visual_related_srcs_att_fa_pooled:(B,NumMix*Obejcts,D)
        sp_att_fv_com_valid:(N,D)
        visual_related_srcs_att_fa_valid:(N,D,F,T)
        -localization
        cos_map:(B,NumMix*Obejcts,H,W)
        sound_localization:(B,NumMix*Obejcts,H,W)
        -out
        av_context_feature:(N,2*D)
        y:(N,2*D,F,T)
        ---------------------
        '''
        outputs={}
        fa_mix=inputs["fa_mix"]
        valid_nums=inputs["valid_nums"]
        fv_common_valid=inputs["fv_common_valid"]#(N,D,H,W)
        _,D,H,W=fv_common_valid.size()
        B,_,F,T=fa_mix.size()
        NumMixObjects=valid_nums.size(1)
        N=fv_common_valid.size(0)
        device=fv_common_valid.device
        outputs["contrast_fa_mix"]=fa_mix[:,:].mean([-1,-2])[:,None,:].repeat(1,NumMixObjects,1)

        ##### separate visual related source feature from mix wav using visual guidance#####
        fv_pooled_repeat = functional.adaptive_max_pool2d(fv_common_valid, [1, 1]).repeat(1, 1, F, T)  # (N,D,F,T)
        fa_mix = flatten_batch(torch.stack([fa_mix] * NumMixObjects, dim=1),valid_nums)  # (N,D,F,T)
        fa=self.sep_mlp(torch.cat([fa_mix,fv_pooled_repeat],dim=1)).view(N,D,F,T)# (N,D,F,T)
        outputs["visual_related_srcs_fa"] = organize_batch(fa,valid_nums)  # (B,NumMixObejcts,D,F,T)

        ##### project visual related source feature to av common space for spatial attention and channel attention#####
        fa_common = self.a_common_mlp(fa).view(N,D,F,T)# (N,D,F,T)
        fa_common=functional.normalize(fa_common,dim=1,p=2)# (N,D,F,T)
        visual_related_srcs_fa_com=functional.normalize(fa_common.mean([-1,-2]),dim=1,p=2)# (N,D)
        outputs["visual_related_srcs_fa_com"]=organize_batch(visual_related_srcs_fa_com,valid_nums) # (B,NumMixObejcts,D)

        ##### A->V spatialattention #####
        if "spatial_attention" in self.opts and not self.opts.spatial_attention:
            sp_att_fv_com=fv_common_valid.max(-1)[0].max(-1)[0]
            outputs["sp_att_fv_com"]=organize_batch(sp_att_fv_com,valid_nums)#(B,NumMixObject,D)
            outputs["sp_att_fv_com_valid"] = sp_att_fv_com#(N,D)
            outputs["cos_map"]=torch.zeros(B,NumMixObjects,H,W).to(device)#(B,NumMixObject,H,W)
            outputs["sound_localization"]=torch.zeros(B,NumMixObjects,H,W).to(device)#(B,NumMixObject,H,W)

        else:
            fa_common1=visual_related_srcs_fa_com.view(N,D,1,1)
            fv_common1=fv_common_valid.view(N,D,H,W)
            cos_map=functional.cosine_similarity(fv_common1,fa_common1,dim=1)#(N,H,W)
            sound_localization=self.sigmoid(cos_map)#(N,H,W)

            sp_att_fv_com=self.weight_pooling(sound_localization,fv_common_valid)#(N,D)
            outputs["sp_att_fv_com"]=organize_batch(sp_att_fv_com,valid_nums)#(B,NumMixObject,D)
            outputs["sp_att_fv_com_valid"] = sp_att_fv_com#(N,D)
            outputs["cos_map"]=organize_batch(cos_map,valid_nums)#(B,NumMixObject,H,W)
            outputs["sound_localization"]=organize_batch(sound_localization,valid_nums)#(B,NumMixObject,H,W)
            # outputs["non_sounding_localization"]=non_sounding_localization#(B,NumMix,1,H,W)

        ##### V->A #####
        #channel attention sp_att_fv_com,fa_common-> visual_related_srcs_att_fa
        if "channel_attention" in self.opts and not self.opts.channel_attention:
            att_fa=fa
            outputs["visual_related_srcs_att_fa"] = organize_batch(att_fa,valid_nums)#(B,NumMixObject,D,F,T)
            outputs["visual_related_srcs_att_fa_valid"] = att_fa#(N,D,F,T)
            outputs["visual_related_srcs_att_fa_pooled"]=outputs["visual_related_srcs_att_fa"].mean([-1,-2])
        else:
            sp_att_fv_com1=sp_att_fv_com.view(N,D,1)
            fa_common2=fa_common.view(N,D,-1)
            f=(sp_att_fv_com1*fa_common2).mean(-1,keepdim=True)#(N,D,1)
            weight=self.weight_mlp(f).view(N,D,1,1)
            att_fa=(weight*fa+fa)
            outputs["channel_attention_weight"] = organize_batch(weight, valid_nums).squeeze()  # (B,NumMixObject,D)
            outputs["visual_related_srcs_att_fa"] = organize_batch(att_fa,valid_nums)#(B,NumMixObject,D,F,T)
            outputs["visual_related_srcs_att_fa_valid"] = att_fa#(N,D,F,T)
            outputs["visual_related_srcs_att_fa_pooled"]=outputs["visual_related_srcs_att_fa"].mean([-1,-2])

        outputs["consistent_feature"]=torch.cat([organize_batch(sp_att_fv_com,valid_nums), outputs["visual_related_srcs_att_fa_pooled"]],
                               dim=2)#(B,NumMixObject,2D)

        ##### to-decode #####
        visual_related_srcs_att_fa1 = att_fa # (N,D,F,T)
        sp_att_fv_com1 = sp_att_fv_com  # (N,D)

        outputs["av_context_feature"]= torch.cat([sp_att_fv_com1, visual_related_srcs_att_fa1.mean([-1, -2])], dim=1)  # (N,2*D)
        outputs["y"] = torch.cat([sp_att_fv_com1[:, :, None, None].repeat(1, 1, F, T), visual_related_srcs_att_fa1],
                               dim=1)  # (N,2*D,F,T)
        return outputs



