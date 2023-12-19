import torch
from torch import nn
from torch.nn import functional
from nnAudio import  features
from utils.utils import RealComplexSpec2MagPhase, warpMagSpec, MagPhase2RealComplexSpec



class StftMelModule(nn.Module):
    def __init__(self,n_fft,hop_length,sr,iSTFT,mel_dim):
        super(StftMelModule, self).__init__()
        self.mel_dim=mel_dim
        self.stftLayer = features.STFT(n_fft=n_fft, hop_length=hop_length, output_format="Complex", sr=sr, freq_scale="no",
                                         iSTFT=iSTFT, trainable=False)  # Initializing the model

    def cal_mel_mag_from_wav(self,audio):
        #B*S->B*F*T
        B,S=audio.size()
        specs=self.stftLayer(audio) # B,F,T,2
        _,F,T,_=specs.size()
        specs=specs.view(B,1,F,T,2)
        mag,phase=RealComplexSpec2MagPhase(specs) #B*1*F*T
        mel_mag=warpMagSpec(mag,True,self.mel_dim).squeeze()
        return mag.squeeze(),mel_mag,phase.squeeze()

    def mix_stft_and_mel_mask_to_sep_audio(self, mel_masks, mix_mag, mix_phase, S):
        #(B,C,F1,T),(B,F2,T),(B,F2,T),int ->#(B,C,S)
        # TODO: 修改
        B, C, F1, T = mel_masks.size()
        F2=mix_mag.size(1)
        stft_masks=warpMagSpec(mel_masks,False,F2)
        est_mags = stft_masks * mix_mag.unsqueeze(dim=1)#(B,C,F,T)


        # 改为采用混合的相位 会有影响吗
        est_phases = torch.stack([mix_phase] * C, dim=1)#(B,C,F,T)
        est_specs = MagPhase2RealComplexSpec(est_mags, est_phases)  # B*C*F*T*2

        est_audios = self.stftLayer.inverse(est_specs.view(B * C, F2, T, 2), onesided=True, length=S)  # B*C,S
        est_audios = est_audios.view(B, C, S)

        return est_audios

    def mix_and_mask_to_sep_audio(self, masks, mix_mag, mix_phase, S):
        #(B,C,F,T),(B,F,T),(B,F,T),int ->#(B,C,S)
        est_mags = masks * mix_mag.unsqueeze(dim=1)#(B,C,F,T)

        B, C, F, T = masks.size()
        # 改为采用混合的相位 会有影响吗
        est_phases = torch.stack([mix_phase] * C, dim=1)#(B,C,F,T)
        est_specs = MagPhase2RealComplexSpec(est_mags, est_phases)  # B*C*F*T*2

        est_audios = self.stftLayer.inverse(est_specs.view(B * C, F, T, 2), onesided=True, length=S)  # B*C,S
        est_audios = est_audios.view(B, C, S)

        return est_audios


    def spec_augment(self,audios,opts):
        # B*S->B*F*T
        #opts: time_mask: n T;   frequency_mask_n F;
        mags,mel_mags,phase=self.cal_mel_mag_from_wav(audios) #B*F*T
        topts=opts.time_mask
        S=audios.size(1)
        B,F1,T=mel_mags.size()
        F2=mags.size(1)
        # if topts.turn_on:
        #     for i in range(topts.n):
        #         t0=(torch.rand(B)*(T-topts.T)).int()
        #         for b in range(B):
        #             mel_mags[b,:,t0[b]:t0[b]+topts.T]=0
        # fopts=opts.frequency_mask
        # if fopts.turn_on:
        #     for i in range(fopts.n):
        #         f0=(torch.rand(B)*(F1-fopts.F)).int()
        #         for b in range(B):
        #             mel_mags[b,f0[b]:f0[b]+fopts.F,:]=0
        # aug_mags=warpMagSpec(mel_mags.unsqueeze(1),False,F2)
        # aug_specs=MagPhase2RealComplexSpec(aug_mags, phase.unsqueeze(1))#
        # aug_audios = self.stftLayer.inverse(aug_specs.squeeze(), onesided=True, length=S)  # B,S

        if topts.turn_on:
            for i in range(topts.n):
                t0=(torch.rand(B)*(T-topts.T)).int()
                for b in range(B):
                    mags[b,:,t0[b]:t0[b]+topts.T]=0
        fopts=opts.frequency_mask
        if fopts.turn_on:
            for i in range(fopts.n):
                f0=(torch.rand(B)*(F2-fopts.F)).int()
                for b in range(B):
                    mags[b,f0[b]:f0[b]+fopts.F,:]=0
        aug_mags=mags
        aug_specs=MagPhase2RealComplexSpec(aug_mags.unsqueeze(1), phase.unsqueeze(1))#
        aug_audios = self.stftLayer.inverse(aug_specs.squeeze(), onesided=True, length=S)  # B,S
        return aug_audios
