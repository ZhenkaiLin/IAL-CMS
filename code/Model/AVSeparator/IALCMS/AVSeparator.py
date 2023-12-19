import torch
from torch import nn
from .DeepLab.visualNet import VisualNet
from .AVModule.AVModule_visual_related_fa_separation import AVModule
from .AVModule.VisualGuidedDisentangleModule import VisualGuidedDisentangleModule
from .SeperateNet.SeparateNet import UnetSeparator as Separator
from utils.utils import *
from utils.utils import sep_val_gpu,sep_val,vis_confusion_matrix,vis_two_mix_ins_separation_array,vis_sdr_volumn_scatter_plot,vis_class_similarity_array,combine_dictionaries
import random
from torch.nn import functional
import pickle
from matplotlib import pyplot as plt
import copy
from .SeperateNet.base_model.StftMelModule import StftMelModule
import pickle

class AVSeparator(nn.Module):
    def __init__(self,opts):
        super().__init__()
        '''
        opts:
            visual_net_opts
            av_module_opts
            separate_net_opts
            normalization: instancenorm
        '''
        self.all_opts=opts
        opts = opts.model.av_separator
        self.opts= opts

        self.visualNet = VisualNet(opts.visual_net)

        avmodule = AVModule(opts.av_module)
        self.separator_w_v = Separator(**opts.separate_net,avmodule=avmodule)

        self.batch_t=0
    @property
    def stftMelModule(self):
        return self.separator_w_v.stftMelModule

    def forward(self,inputs):
        '''
        -------Inputs------
        imgs:B,NumMix*Objects,3,H,W
        mixed_mixture:  B,AudLen
        valid_num:(B,NumMix*Objects)
        -------------------

        -------Outputs------
        -SepResults
        est_components:B,NumMix*Objects,AudLen
        est_score_map,est_mel_mask,est_mel_mag:B,NumMix*Objects,F,T
        -LocResults
        cos_map:(B,NumMix*Objects,H,W)
        sound_localization:(B,NumMix*Objects,H,W)
        non_sounding_localization:(B,NumMix*Objects,1,H,W)
        -LeanedRepresentation
        fv:(B,NumMix*Objects,D,H,W)
        visual_related_srcs_fa_com:(B,NumMix*Objects,D)
        sp_att_fv_com:(B,NumMix*Objects,D)
        visual_related_srcs_att_fa:(B,NumMix*Objects,D,F,T)
        visual_related_srcs_att_fa_pooled:(B,NumMix*Objects,D)
        -------------------
        '''
        valid_nums=inputs["valid_nums"]
        imgs=inputs["imgs"]
        valid_imgs=flatten_batch(imgs,valid_nums)
        fv_valid,dropout_common_fv_valid=self.visualNet(valid_imgs)
        fv=organize_batch(fv_valid,valid_nums)
        dropout_common_fv=organize_batch(dropout_common_fv_valid,valid_nums)

        separator_inputs=dict({"fv":fv,"fv_valid":fv_valid},**inputs)
        separator_outputs=self.separator_w_v(separator_inputs)

        return dict({"fv":fv,"dropout_common_fv":dropout_common_fv},**separator_outputs)

    def calculate_loss(self,inputs):
        losses=[]
        log_input={}
        loss_opt=self.opts.loss

        sep_loss_input = dict(
            inputs, **loss_opt.sep_loss_opt)
        sep_loss = self.sep_loss(sep_loss_input)
        losses.append(sep_loss)
        log_input["sep_loss"] = sep_loss

        if not loss_opt.lambda_avc == 0:
            avc_loss_input = dict(
                inputs, **{"opts": self.opts.loss.avc_contrastive})
            avc_loss = self.contrastive_loss(avc_loss_input)
            losses.append(loss_opt.lambda_avc * avc_loss)

            log_input["avc_loss"] = avc_loss
            log_input["lambda_avc"] = loss_opt.lambda_avc

        if str(self.batch_t) in loss_opt.lambda_avc_incresing_step:
            loss_opt.lambda_avc *= 10
        self.batch_t += 1

        return torch.stack(losses).sum(),log_input

    def sep_loss(self,inputs):
        '''
        -------Inputs------
        mixtures:B,NumMix,AudLen
        **SepOutputs
        -SepLossOpts
            loss_type(spec_loss,mask_loss,wav_loss)
            max_sisnr
        -------------------

        -------Outputs------
        sep_loss
        -------------------
        '''

        mixed_mixture=inputs["mixed_mixture"]
        est_components = inputs["est_components"]
        mixtures = inputs["mixtures"]
        mask=inputs["components_valid_nums"][:,:,:,None]
        est_mixtures = (est_components*mask).sum(2)  # (B,NumMix,AudLen)
        est_mel_mask=inputs["est_mel_mask"]
        if inputs["sep_loss_type"]=="wav":
            if inputs["max_sisnr"]=="inf":
                sisnr = cal_video_sisnr(mixtures, est_mixtures)
            else:
                sisnr = cal_video_sisnr(mixtures, est_mixtures,inputs["max_sisnr"])
            return -sisnr.mean()
        else:
            raise AttributeError("Unkown SepLoss Type")
    def gt_mel_mask(self,mixed_mixture, mixtures):
        return self.separator_w_v.gt_mel_mask(mixed_mixture, mixtures)


    def contrastive_loss(self,inputs):
        '''
        -------Inputs------
        dropout_common_fv:(B,NumMix*Objects,D,H,W)
        visual_related_srcs_fa_com:(B,NumMix,Objects,D)
        components_valid_nums: (B,NumMix,Objects)
        train_background_input:
        opts:
            t: temparature control bounding,hardness aware, local separation and others
            hard_sampling:
                discard_K: alleviate the problem of false negative
            positive_weight:
                turn_on: try to add hard sampling property on (anchor,{pos},{neg}) selection like negative hard_sampling
                t: controling the strenghth of hard sampling property
            dropout:
                use dropout on visual feature map alleviate overfitting and mining more positive patch on Image
        -------------------

        -------Outputs------
        avc_loss
        -------------------
        '''
        opts=inputs["opts"]
        common_fv=inputs["dropout_common_fv"]

        if opts.contrasted_feature=="fusion":
            contasted_feature = inputs["visual_related_srcs_fa_com"]
            contasted_feature=contasted_feature.flatten(end_dim=2)
            #assert C==1


        B,NumMix,Object,D,H,W=common_fv.size()
        NumMixObject=NumMix*Object
        device=common_fv.device
        N=contasted_feature.size(0)

        components_valid_nums = inputs["components_valid_nums"]
        components_valid_nums=components_valid_nums.flatten().bool()

        common_fv=common_fv.view(N,D,H,W)[components_valid_nums]
        common_fv=common_fv.unsqueeze(1)
        contasted_feature=contasted_feature.view(1,N,D,1,1)[:,components_valid_nums]
        Valid_N=common_fv.size(0)
        cos_array = functional.cosine_similarity(common_fv, contasted_feature, dim=2).max(-1)[0].max(-1)[0]#(N,N)

        O_pos = torch.eye(Valid_N).to(device)
        O_neg = torch.logical_not(O_pos)

        return 0.5 * (
                cal_Semantic_Tolerence_DML(cos_array, O_pos, O_neg, opts) +
                cal_Semantic_Tolerence_DML(cos_array.permute(1, 0),
                                           O_pos.permute(1, 0), O_neg.permute(1, 0)
                                           , opts))
    def on_validation_start(self):
        pass

    def validation_step(self,validate_input):
        return {}

    def get_opt_dict(self,opts):
        return [{"params":self.separator_w_v.parameters(),"lr":opts.a_lr},{"params":self.visualNet.parameters(),"lr":opts.v_lr}]


