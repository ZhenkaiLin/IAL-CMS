from torch.nn import functional
from .base_model.StftMelModule import StftMelModule

import torch
import torch.nn as nn
from functools import partial
from utils.utils import organize_batch,flatten_batch
import random
def conv3x3(in_dim, out_dim, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_dim, out_dim, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, indim, dim, stride=1, downsample=None, norm_layer=None,relu=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(indim, dim, stride)
        self.bn1 = norm_layer(dim)
        self.relu = relu(inplace=True)
        self.conv2 = conv3x3(dim, dim)
        self.bn2 = norm_layer(dim)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class UnetSeparator(nn.Module):
    def __init__(self,n_fft, hop_length, sr,mel_dim,mask,avmodule,use_av_module,multi_scale_avc,dense_visual_guidance,resnet_encoder=True,**kwargs):
        super(UnetSeparator, self).__init__()
        self.resnet_encoder=resnet_encoder
        self.activation_function=partial(nn.LeakyReLU, negative_slope=0.2)
        self._norm_layer=nn.BatchNorm2d
        #use_dropout = False
        self.stftMelModule = StftMelModule(n_fft=n_fft, hop_length=hop_length, sr=sr, iSTFT=True,mel_dim=mel_dim)
        if avmodule.opts.disentanglement:
            av_context_dim = avmodule.opts.v_in_dim
        else:
            av_context_dim=avmodule.opts.v_in_dim*2
        if resnet_encoder:
            encoder_dims = [64, 64, 128, 256, 512]
        else:
            encoder_dims=[64,128,256,512,512]

        unet_block = UnetBlock(None,
            input_nc=512, inner_input_nc=None,inner_output_nc=None, outer_nc=1024,
            submodule=None, innermost=True,avmodule=avmodule,
            av_context_dim=av_context_dim,use_av_module=use_av_module,
           multi_scale_avc=multi_scale_avc,dense_visual_guidance=dense_visual_guidance,
                               multi_scale_fa_dim=960)
        # not use short connect because avmodule don't downsample feature
        unet_block = UnetBlock(self._make_encoder_block(indim=encoder_dims[3],dim=encoder_dims[4],blocks=2,stride=2),
            input_nc=256, inner_input_nc=encoder_dims[4],inner_output_nc=1024, outer_nc=512,
            submodule=unet_block,av_context_dim=av_context_dim,use_av_module=use_av_module,use_short_connect=False,
            multi_scale_avc=multi_scale_avc,dense_visual_guidance=dense_visual_guidance)
        unet_block = UnetBlock(self._make_encoder_block(indim=encoder_dims[2],dim=encoder_dims[3],blocks=2,stride=2),
            input_nc=128, inner_input_nc=encoder_dims[3],inner_output_nc=512, outer_nc=256,
            submodule=unet_block,av_context_dim=av_context_dim,use_av_module=use_av_module,
                               multi_scale_avc=multi_scale_avc,dense_visual_guidance=dense_visual_guidance)
        unet_block = UnetBlock(self._make_encoder_block(indim=encoder_dims[1],dim=encoder_dims[2],blocks=2,stride=2),
            input_nc=64, inner_input_nc=encoder_dims[2],inner_output_nc=256, outer_nc=128,
            submodule=unet_block,av_context_dim=av_context_dim,use_av_module=use_av_module,
                               multi_scale_avc=multi_scale_avc,dense_visual_guidance=dense_visual_guidance)
        unet_block = UnetBlock(self._make_encoder_block(indim=encoder_dims[0],dim=encoder_dims[1],blocks=2,stride=2),
            input_nc=64, inner_input_nc=encoder_dims[1],inner_output_nc=128, outer_nc=64,
            submodule=unet_block,av_context_dim=av_context_dim,use_av_module=use_av_module,
                               multi_scale_avc=multi_scale_avc,dense_visual_guidance=dense_visual_guidance)
        unet_block = UnetBlock(self._first_encoder_block(),
            input_nc=1, inner_input_nc=encoder_dims[0],inner_output_nc=64, outer_nc=1,
            submodule=unet_block, outermost=True,av_context_dim=av_context_dim,use_av_module=use_av_module,
                               multi_scale_avc=multi_scale_avc,dense_visual_guidance=dense_visual_guidance)

        self.mask=mask
        self.bn0 = self._norm_layer(1)
        self.unet_block = unet_block
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.kwargs=kwargs

    def get_av_module(self):
        return self.unet_block.submodule.submodule.submodule.submodule.submodule.avmodule

    def _first_encoder_block(self):
        conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        bn = nn.BatchNorm2d(64)
        relu = self.activation_function(inplace=True)

        return nn.Sequential(conv,bn,relu)

    def _make_encoder_block(self,indim, dim, blocks, stride=1):
        if self.resnet_encoder:
            norm_layer = self._norm_layer
            downsample = None
            if stride != 1:
                downsample = nn.Sequential(
                    conv1x1(indim, dim , stride),
                    norm_layer(dim),
                )

            layers = []
            layers.append(BasicBlock(indim, dim, stride, downsample, norm_layer,relu=self.activation_function))
            indim = dim
            for _ in range(1, blocks):
                layers.append(BasicBlock(indim, dim,norm_layer=norm_layer,relu=self.activation_function))

            return nn.Sequential(*layers)
        else:
            downconv = nn.Conv2d(indim, dim, kernel_size=4, stride=2, padding=1)
            downrelu = nn.LeakyReLU(0.2, True)
            downnorm = nn.BatchNorm2d(dim)
            return nn.Sequential(*[downconv, downnorm, downrelu])

    def forward(self, inputs):
        '''
        ------- Input ------
        mixed_mixture:(B,AudLen)
        fv_valid:(N,D,H1,W1)
        valid_nums:(B,NumMix*Objects)
        --------------------

        ------- Output ------
        -SepResults
        est_components:B,NumMix*Obejcts,AudLen
        est_score_map,est_mel_mask,est_mel_mag:B,NumMix*Obejcts,F,T
        -LeanedRepresentation
        visual_related_srcs_fa_com:(B,NumMix*Obejcts,D)
        sp_att_fv_com:(B,NumMix*Obejcts,D)
        visual_related_srcs_att_fa:(B,NumMix*Obejcts,D,F,T)
        visual_related_srcs_att_fa_pooled:(B,NumMix*Obejcts,D)
        -Localization
        cos_map:(B,NumMix*Obejcts,H,W)
        sound_localization:(B,NumMix*Obejcts,H,W)
        ---------------------

        '''
        mixed_mixture, fv,fv_valid=[inputs["mixed_mixture"],inputs["fv"],inputs["fv_valid"]]
        B,AudLen=mixed_mixture.size()
        NumMixObject=fv.size(1)
        mix_mag,mix_mel_mag,mix_phase=self.stftMelModule.cal_mel_mag_from_wav(mixed_mixture) #(B,F,T)
        _,F,T=mix_mel_mag.size()
        mix_mel_mag=mix_mel_mag.unsqueeze(1)
        log_mix_mel_mag=torch.log(mix_mel_mag+1e-8).detach()

        x = self.bn0(log_mix_mel_mag)# (B,1,F,T)
        score_map,outputs= self.unet_block(x,fv_valid,inputs) # (N,1,F,T)
        score_map=organize_batch(score_map.view(-1,F,T),inputs["valid_nums"])# (B,NumMixObject,F,T)

        score_map=score_map*self.scale+self.bias
        if self.mask == "IBM":
            est_mel_mask = torch.sigmoid(score_map)
        elif self.mask == "IRM":
            if self.kwargs.get("IRM_bias",None):
                score_map=score_map+0.5
            est_mel_mask = functional.relu(score_map)
        else:
            raise AttributeError("Unknown Mask Type.")
        est_mel_mag =est_mel_mask*mix_mel_mag
        est_wav = self.stftMelModule.mix_stft_and_mel_mask_to_sep_audio(est_mel_mask,mix_mag,
                                                                        mix_phase, AudLen)#(B, -1, AudLen)

        outputs=dict({
            "est_components":est_wav.view(B,NumMixObject,AudLen),
            "est_score_map":score_map.view(B,NumMixObject,F,T),
            "est_mel_mask":est_mel_mask.view(B,NumMixObject,F,T),
            "est_mel_mag":est_mel_mag.view(B,NumMixObject,F,T)
            },**outputs)

        return outputs


    def gt_mel_mask(self, mixed_mixture, audios):
        # (B,S) (B,C,S) -> (B,C,F,T)
        B,C,S=audios.size()

        _,mel_mag_mix,_=self.stftMelModule.cal_mel_mag_from_wav(mixed_mixture)
        _, F, T = mel_mag_mix.size()
        mel_mag_mix=mel_mag_mix.unsqueeze(1)
        _, mel_mags, _ = self.stftMelModule.cal_mel_mag_from_wav(audios.flatten(end_dim=1))
        mel_mags=mel_mags.view(B,C,F,T)
        if self.mask == "IBM":
            gt_mel_masks = (mel_mags > 0.5 * mel_mag_mix).type_as(mel_mag_mix)
        if self.mask == "IRM":
            gt_mel_masks = mel_mags / (mel_mag_mix+1e-8)
            gt_mel_masks = torch.clip(gt_mel_masks, 0, 5)
        weight = torch.log1p(mel_mag_mix)
        weight = torch.clamp(weight, 1e-3, 10)
        return gt_mel_masks.detach(),mel_mag_mix,mel_mags,weight



# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(self, encoder,outer_nc=None, inner_input_nc=None, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 use_dropout=False, inner_output_nc=None, noskip=False,
                 innnermost_args=None,av_context_dim=1024,avmodule=None
                 ,use_av_module=True,use_short_connect=True,
                 multi_scale_avc=False,dense_visual_guidance=True,
                 multi_scale_fa_dim=None):
        super(UnetBlock, self).__init__()
        self.use_av_module=use_av_module
        self.outermost = outermost
        self.noskip = noskip
        self.innermost=innermost

        self.multi_scale_avc=multi_scale_avc
        self.dense_visual_guidance=dense_visual_guidance

        use_bias = False
        if input_nc is None:
            input_nc = outer_nc
        if inner_output_nc==None:
            inner_output_nc = inner_input_nc

        uprelu = nn.LeakyReLU(0.2, True)
        upnorm = nn.BatchNorm2d(outer_nc)
        upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        if innermost:
            self.avmodule=avmodule
        else:
            if outermost:
                upconv = nn.Conv2d(
                    inner_input_nc+inner_output_nc, outer_nc, kernel_size=3, padding=1)
                up = [upsample,upconv]
            else:
                if use_short_connect:
                    up_in_dim=inner_input_nc+inner_output_nc
                else:
                    up_in_dim = inner_output_nc
                upconv = nn.Conv2d(
                    up_in_dim, outer_nc, kernel_size=3,
                    padding=1, bias=use_bias)
                up = [upsample,upconv,uprelu,upnorm]
                if self.dense_visual_guidance == "AdaIN":
                    self.instance_norm=nn.InstanceNorm2d(outer_nc,affine=False,track_running_stats=False)
                    self.style_mlp_w=nn.Sequential(nn.Linear(av_context_dim,outer_nc))
                    self.style_mlp_b = nn.Sequential(nn.Linear(av_context_dim, outer_nc))

            self.down = encoder
            self.submodule=submodule
            self.up=nn.Sequential(*up)

    def duplicate(self,x,n):
        return torch.stack([x]*n,dim=1).flatten(end_dim=1)

    def forward(self, x,fv,inputs):
        # print(x.shape)
        valid_nums=inputs["valid_nums"]
        if self.innermost:
            B, _, H, W = x.size()
            C=self.avmodule.opts.max_source

            avmodule_input=dict(inputs,**{"fv_common_valid":fv,"fa_mix":x})
            avmodule_outputs=self.avmodule(avmodule_input)
            '''
            ------- AVModule_Output ------
            -extrated_features
            visual_related_srcs_fa_com:(B,NumMix*Obejcts,C,D)
            sp_att_fv_com:(B,NumMix*Obejcts,C,D)
            visual_related_srcs_att_fa:(B,NumMix*Obejcts,C,D,F,T)
            visual_related_srcs_att_fa_pooled:(B,NumMix*Obejcts,C,D)
            sp_att_fv_com_valid:(N,C,D)
            visual_related_srcs_att_fa_valid:(N,C,D,F,T)
            
            -localization
            cos_map:(B,NumMix*Obejcts,C,H,W)
            sound_localization:(B,NumMix*Obejcts,C,H,W)
            ---------------------
            '''
            y=avmodule_outputs["y"]
            return y,avmodule_outputs["av_context_feature"],avmodule_outputs

        else:
            if self.outermost:
                y=self.down(x)
                y,av_context_feature,outputs=self.submodule(y,fv,inputs)
                y=self.up(y) # (N*C,1)
                return y,outputs
            else:
                y = self.down(x)
                y,av_context_feature,outputs=self.submodule(y,fv,inputs)
                y = self.up(y)

                if self.dense_visual_guidance=="AdaIN":
                    w=self.style_mlp_w(av_context_feature)[:, :, None, None] # (N*C,D,1,1)
                    b=self.style_mlp_b(av_context_feature)[:, :, None, None] # (N*C,D,1,1)
                    y=self.instance_norm(y)
                    y=y*w+b

                d_x = flatten_batch(torch.stack([x]*valid_nums.size(1),dim=1),valid_nums)

                return torch.cat([d_x,y],1),av_context_feature,outputs

