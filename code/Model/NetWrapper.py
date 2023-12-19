import os

import torch

from Model.AVSeparator.IALCMS.AVSeparator import AVSeparator as IAL_AVSeparator
from Model.Critics.critics_manager import CriticsManager
from utils.utils import *
from os.path import join
from functools import partial
from utils.HTMLvisualizer import HTMLVisualizer
from torch import nn
import os
class NetWrapper(nn.Module):
    def __init__(self,opts):
        super().__init__()
        mopts = opts.model
        self.critics=CriticsManager().configCritics(mopts.critics)

        self.av_separator=self.select_algorithm(opts)

    @property
    def stftMelModule(self):
        return self.av_separator.stftMelModule

    def select_algorithm(self,opts):
        if opts.algorithm == "IAL-CMS":
            AVSeparator = IAL_AVSeparator
        return AVSeparator(opts)

    def visualize(self,visualize_input):
        '''
        --------- inputs ---------
        vis_dir:
        support visualization:
        visualize_av_separate_results:
            -SepResults
            est_components:B,NumMix,Objects*C,AudLen
            est_mel_mask,est_mel_mag:B,NumMix,C,F,T
            -LocResults
            cos_map:(B,NumMix,Objects*C,H,W)
            sound_localization:(B,NumMix,Objects*C,H,W)
            non_sounding_localization:(B,NumMix,1,H,W)
        "visualize_IAL_gradients":critics_output["gradients"],
        --------------------------
        '''
        vis_dir=visualize_input["vis_dir"]
        os.makedirs(vis_dir,exist_ok=True)
        self.visualize_av_separate_results(dict({"vis_dir":vis_dir},**visualize_input["visualize_av_separate_results"]))


    def get_mel_mag(self, wavs):
        _, mel_mag_mix, _ = self.stftMelModule.cal_mel_mag_from_wav(wavs)
        return mel_mag_mix

    def gt_mel_mask(self, mixed_mixture, audios):
        # (B,S) (B,C,S) -> (B,C,F,T)
        return self.av_separator.gt_mel_mask(mixed_mixture, audios)

    def visualize_av_separate_results(self,inputs):
        '''
        --------- inputs ---------
        vis_dir
        paths:optional
        valid_nums:B,NumMix*Objects
        -SepInputs
        imgs(objects):B,NumMix*Objects,3,H,W
        mixed_mixture:  B,AudLen
        -SepResults
        est_components:B,NumMix,Objects*C,AudLen
        est_mel_mask,est_mel_mag:B,NumMix,Objects*C,F,T
        -GroundTruth
        mixtures:B,NumMix,AudLen
        (optional)
        -LocResults
        cos_map:(B,NumMix,Objects*C,H,W)
        sound_localization:(B,NumMix,Objects*C,H,W)

        --------------------------
        '''

        vis_dir = inputs["vis_dir"]
        html_visualizer = HTMLVisualizer(join(vis_dir,"av_separation_visualization.html"))

        if "paths" in inputs:
            with open(join(vis_dir,"path.csv"),"w") as f:
                sdrs=inputs["sdr"]#(B,NumMix)
                for item,sdr in zip(inputs["paths"],sdrs):
                    sdr=[str(float(value)) for value in sdr]
                    f.write(",".join(item+sdr) + '\n')

        est_mel_mask = inputs["est_mel_mask"]
        est_components=inputs["est_components"]
        mixtures=inputs["mixtures"]
        mixed_mixture = inputs["mixed_mixture"]
        est_mel_mag = inputs["est_mel_mag"]
        img=inputs["imgs"]

        # separation performance is irrealted to volumn
        est_mixture= est_components.sum(2)
        volumn_ratio=torch.norm(mixtures,dim=2,p=2)/(torch.norm(est_mixture,dim=2,p=2)+1e-8)#(B,NumMix)
        est_mel_mag*=volumn_ratio[:,:,None,None,None]
        est_components*=volumn_ratio[:,:,None,None]
        est_mel_mask*=volumn_ratio[:,:,None,None,None]

        B, NumMix,MaxObjectC,AudLen =est_components.size()
        MaxObject=img.size(1)//NumMix
        _,_,_,H,W=img.size()
        C=MaxObjectC//MaxObject
        objects=img.view(B,NumMix,MaxObject,3,H,W)
        _,_,_,F,T=est_mel_mag.size()
        device=est_mel_mag.device
        valid_nums = inputs["valid_nums"].view(B,NumMix,MaxObject)
        vis_select_mask=torch.ones(B,NumMix,MaxObjectC).to(device)
        gt_mel_mag=self.get_mel_mag(mixtures.flatten(end_dim=1)).view(B,NumMix,F,T)#(B,NumMix,F,T)
        mix_mel_mag=self.get_mel_mag(mixed_mixture)#(B,NumMix,F,T)
        # mix_mel_mask, _,_,_ = self.gt_mel_mask(mixed_mixture, mixtures)  # (B,NumMix,F,T)
        vis_mel_mag=vis_spectrogram
        for b in range(B):
            namef1 = partial(vis_name, vis_dir=vis_dir, b=b)
            pathf1=partial(pathf, b=b)
            ###visualize mixed_mixture
            vis_audio(mixed_mixture[b], namef1(nm=0,sub1="", sub2="mix_wav"))
            vis_mel_mag(mix_mel_mag[b], namef1(nm=0,sub1="", sub2="mix_wav"))
            mixed_mixture_path = [pathf1(nm=0, sub1="", sub2="mix_wav")+".png"]
            b_mix_mel_mag=mix_mel_mag[b]
            video_frame_path=[]
            gt_audios_path=[]
            separated_sources_path=[]
            sources_loc_path=[]
            selected=vis_select_mask[b].bool().flatten()
            est_video_audio_path=[]
            for nm in range(NumMix):
                vis_f = partial(vis_loc_or_segment, vis_dir=vis_dir, b=b, nm=nm)
                namef = partial(vis_name, vis_dir=vis_dir, b=b, nm=nm)
                pathf1=partial(pathf1,nm=nm)
                vis_audio(mixtures[b,nm],namef(sub1="",sub2="gt_wav"))
                vis_mel_mag(gt_mel_mag[b,nm],namef(sub1="",sub2="gt_wav"))
                gt_audios_path.append(pathf1(sub1="", sub2="gt_wav")+".png")
                est_video_mel_mag=((b_mix_mel_mag.unsqueeze(0)*est_mel_mask[b,nm])*vis_select_mask[b,nm][:,None,None]).sum(0)
                vis_mel_mag(est_video_mel_mag,namef(sub1="", sub2="est_video_mel_mag"))
                est_video_audio_path.append(pathf1(sub1="", sub2="est_video_mel_mag")+".png")
                for o in range(MaxObject):
                    if valid_nums[b,nm,o]:
                        vis_image(objects[b,nm,o],namef(sub1="object%d"%o,sub2="frame"))
                        video_frame_path.append(pathf1(sub1="object%d"%o, sub2="frame") + ".jpg")
                        if "sound_localization" in inputs and "cos_map" in inputs:
                            for c in range(C):
                                vis_f(img=objects[b,nm, o],cam=inputs["sound_localization"][b,nm, C*o+c],sub1="obj%d_src%d"%(o,c),sub2="ssl")
                                vis_f(img=objects[b,nm, o],cam=inputs["cos_map"][b, nm, C*o+c],sub1="obj%d_src%d"%(o,c), sub2="cos_map")
                for mc in range(MaxObjectC):
                    ###visualize separated visual-related components and localization results
                    if valid_nums[b, nm, mc//C]:
                        sources_loc_path.append(pathf1(sub1="src%03d" % mc, sub2="cos_map")+".jpg")
                        vis_mel_mag(est_mel_mag[b, nm, mc], namef(sub1="src%03d" % mc,sub2="est_mel_mag"))
                        separated_sources_path.append(pathf1(sub1="src%03d" % mc,sub2="est_mel_mag")+".png")
                        vis_audio(est_components[b, nm, mc],namef(sub1="src%03d" % mc, sub2="est_components"))
                        # binary_mask=est_mel_mask[b,nm,mc]
                        # binary_mask[binary_mask>0.5]=1
                        # binary_mask[binary_mask <0.5] = 0
                        vis_mask(est_mel_mask[b,nm,mc],namef(sub1="src%03d" % mc, sub2="est_mask"),vmax=2,vmin=0)
            html_visualizer.add_content(b, mixed_mixture_path,video_frame_path,gt_audios_path,separated_sources_path,sources_loc_path,selected,est_video_audio_path)
        html_visualizer.write_html()


    def validate(self,inputs):
        '''
        --------- inputs ---------
        -SepResults
        est_components:B,NumMic,C,AudLen
        -GroundTruth
        mixtures:B,NumMix,AudLen
        -Inputs
        mixed_mixture: B,AudLen
        --------------------------
        '''
        est_components=inputs["est_components"]
        mixtures=inputs["mixtures"]
        est_mixtures = est_components.sum(2)  # (B,NumMix,AudLen)
        result,record_result= sep_val(mixtures, est_mixtures,est_mixtures.size(0))
        sdr,sir,sar=result
        outputs={"metrics":{"sdr":sdr,"sir":sir,"sar":sar}
            ,"result_per_sample":record_result}

        # if "mixed_mixture" in inputs:
        #     mixed_mixture=inputs["mixed_mixture"]
        #     mixed_mixture=torch.stack([mixed_mixture]*mixtures.size(1),dim=1)
        #     result_mix, record_result_mix = sep_val(mixtures, mixed_mixture, mixed_mixture.size(0))
        #     outputs["metrics"]["sdri"]=sdr-result_mix[0]
        #     outputs["metrics"]["siri"] = sir - result_mix[1]
        return outputs