import copy

import pytorch_lightning  as pl
import pytorch_lightning.utilities
import torch
from Model.NetWrapper import NetWrapper
from easydict import  EasyDict
import numpy as np
from os.path import join
import sys
from utils.utils import sep_val_gpu,sep_val,vis_confusion_matrix,vis_two_mix_ins_separation_array,vis_sdr_volumn_scatter_plot,\
    vis_class_similarity_array,combine_dictionaries,SIR_calculation,SAR_calculation
import random
from torch.nn import functional
import pickle
from matplotlib import pyplot as plt
import random
from itertools import accumulate
from functools import partial


def warm_up_lambda(step,increase_steps):
    i=0
    while i<len(increase_steps):
        if step<increase_steps[i]:
            break
        i+=1
    return 1/10**(len(increase_steps)-i)


class AdversarialTrainLM(pl.LightningModule):
    def __init__(self,opts):
        super(AdversarialTrainLM, self).__init__()
        self.opts=opts
        self.net_wrapper=NetWrapper(self.opts)

        self.automatic_optimization = False
        #### log variable####
        self.best_metric=-999
        self.epoch_t=0
        self.batch_t=0
        self.net_wrapper.av_separator.logger=self.logger

    def configure_optimizers(self):
        oopts=self.opts.optimizer
        gopts=oopts.av_separator
        if gopts.algorithm=="Adam":
            self.optimizer_G = torch.optim.Adam(self.net_wrapper.av_separator.get_opt_dict(gopts),betas=(gopts.b1, gopts.b2),weight_decay=gopts.weight_decay)

        if gopts.lr_adjust_strategy:
            las_opt=gopts.lr_adjust_strategy
            if las_opt.algorithm=="WarmUp":
                self.lr_scheduler_G=torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer_G,lr_lambda=partial(warm_up_lambda,increase_steps=las_opt.steps),verbose=True)

        else:
            self.lr_scheduler_G=None

        copts=oopts.critics
        self.optimizer_C = torch.optim.Adam(self.net_wrapper.critics.parameters(), lr=copts.lr, betas=(copts.b1, copts.b2),weight_decay=copts.weight_decay)
        return None

    def get_av_module(self):
        return self.net_wrapper.av_separator.separator_w_v.get_av_module()

    def on_validation_epoch_end(self):
        self.epoch_t+=1

    def pl_log(self,data_dict,val=False):
        prefix="hp/" if val else ""
        if val:
            for key,value in data_dict.items():
                self.log(prefix+key, value, prog_bar=True, logger=True,sync_dist=True)
        else:
            for key,value in data_dict.items():
                self.log(prefix+key, value, prog_bar=True, logger=True)

    def av_sep_forward(self,sep_input,Objects,separator):
        '''

        -------Outputs------
        -SepResults
        est_components:B,NumMix*Objects,AudLen
        est_score_map,est_mel_mask,est_mel_mag:B,NumMix*Objects,F,T
        -LocResults
        cos_map:(B,NumMix*Objects,H,W)
        sound_localization:(B,NumMix*Objects,H,W)
        -LeanedRepresentation
        fv:(B,NumMix*Objects,D,H,W)
        visual_related_srcs_fa_com:(B,NumMix*Objects,D)
        sp_att_fv_com:(B,NumMix*Objects,D)
        visual_related_srcs_att_fa:(B,NumMix*Objects,D,F,T)
        visual_related_srcs_att_fa_pooled:(B,NumMix*Objects,D)
        -------------------
        '''
        #B,NumMix*Objects -> B,NumMix,Obj
        sep_input["valid_nums"]=sep_input["valid_nums"].flatten(start_dim=1, end_dim=2)
        sep_output = separator(sep_input)
        B,NumMixObjects,_=sep_output["est_components"].size()
        NumMix=NumMixObjects//Objects
        sep_output["components_valid_nums"] = sep_input["valid_nums"]
        for name,value in sep_output.items():
            if type(value)==torch.Tensor and len(value.size())>=2 and value.size(1)==NumMixObjects:
                sep_output[name]=value.view(B,NumMix,Objects,*value.size()[2:])
        return sep_output

    def log_critics_train(self,critics_output,log_input):
        log_input["c_est_dist_d"] = critics_output["est_dist_d"]
        return log_input

    def on_train_epoch_end(self) -> None:
        if not self.lr_scheduler_G== None:
            self.lr_scheduler_G.step()

    def on_fit_start(self) -> None:
        self.net_wrapper.critics.critics_model.pair_sample_module.memory_bank.memory_bank = self.net_wrapper.critics.critics_model.pair_sample_module.memory_bank.memory_bank.to(
            "cpu")
    def training_step(self,batch,batchid):
        '''
        batch:
            mixed_mixture: (B,AudLen)
            mixtures: (B,NumMix,AudLen)
            objects:(B,NumMix,Objects,3,H,W)
            frames: (B,NumMix,3,H,W)
            backgrounds:(B,NumMix,3,H,W)
            valid_nums:(B,NumMix,Objects)
            labels: (B,NumMix,Objects)
        '''
        Objects = batch["objects"].size(2)

        log_input={}
        if self.opts.volumn_normalize:
            volumn = torch.norm(batch["mixed_mixture"], dim=1)
            if "mean_volumn" in self.opts:
                volumn = volumn / self.opts.mean_volumn
            batch["mixed_mixture"]=batch["mixed_mixture"]/(volumn[:,None]+1e-8)
            batch["mixtures"] = batch["mixtures"]/(volumn[:,None,None]+1e-8)

        if batchid % self.opts.train_critics_interval == 0:
            # ---------------------
            #  Train Critics
            # ---------------------
            if not self.opts.lambda_ind==0:
                with torch.no_grad():
                    # Sep components from mixed_mixture using visual guidance

                    sep_input = {"imgs": batch["objects"].flatten(start_dim=1, end_dim=2),"frames":batch["frames"],
                                 "mixed_mixture": batch["mixed_mixture"],"valid_nums":batch["valid_nums"],"mixtures": batch["mixtures"]}
                    sep_output=self.av_sep_forward(sep_input,Objects,self.net_wrapper.av_separator)

                # Construct independent components pair and dependent components pairs
                critics_input= {"mixtures":batch["mixtures"],"mixed_mixtures":batch["mixed_mixture"],**sep_output}
                critics_output=self.net_wrapper.critics.critics_training_forward(critics_input)
                # Critics loss
                c_loss=critics_output["c_loss"]
                # Optimize Critics to estimate wasserstein distance between independent and dependent distribution (Estimating Mutual Information)

                self.optimizer_C.zero_grad()
                c_loss.backward()
                self.optimizer_C.step()

                # log critics training
                log_input["c_loss"] = c_loss
                log_input=self.log_critics_train(critics_output,log_input)

        if batchid % self.opts.train_generator_interval==0:
            # ---------------------
            #  Train Generator
            # ---------------------
            losses=[]
            sep_input = {"imgs": batch["objects"].flatten(start_dim=1, end_dim=2),"frames":batch["frames"],
                         "mixed_mixture": batch["mixed_mixture"],"valid_nums":batch["valid_nums"]}
            sep_output=self.av_sep_forward(sep_input,Objects,self.net_wrapper.av_separator)

            separator_loss_input={"mixtures": batch["mixtures"], "mixed_mixture": batch["mixed_mixture"], "labels": batch["pseudo_labels"],**sep_output}
            separator_loss,separator_log=self.net_wrapper.av_separator.calculate_loss(separator_loss_input)
            losses.append(separator_loss)
            log_input.update(separator_log)
            if not self.opts.lambda_ind == 0:
                critics_input= {"mixtures":batch["mixtures"],"mixed_mixtures":batch["mixed_mixture"],**sep_output}
                critics_output = self.net_wrapper.critics.generator_training_forward(critics_input)

                ind_loss = critics_output["g_loss"]
                losses.append(self.opts.lambda_ind * ind_loss)
                log_input["lambda_ind"] = self.opts.lambda_ind
                log_input["ind_loss"] = ind_loss
                # log_input["ind_gardients_norm_on_components"] = critics_output["gradients_norm"]
                log_input["g_est_dist_d"] = critics_output["est_dist_d"]

            total_g_loss=torch.stack(losses).sum()
            self.optimizer_G.zero_grad()
            total_g_loss.backward()
            self.optimizer_G.step()
            log_input["total_g_loss"] = total_g_loss

            if self.opts.volumn_normalize:
                sep_output["est_components"] = sep_output["est_components"] * (volumn[:, None, None, None])
                sep_output["est_mel_mag"] = sep_output["est_mel_mag"] * (volumn[:, None, None, None, None])
                batch["mixed_mixture"] = batch["mixed_mixture"] * (volumn[:, None])
                batch["mixtures"] = batch["mixtures"] * (volumn[:, None, None])
            # visualize separation
            if batchid % self.opts.n_visualize==0 and self.opts.exp_vis_root and self.global_rank==0:
                visualize_input = {"vis_dir":join(self.opts.exp_vis_root,"Visualizations","batch%d"%self.batch_t),
                                   "visualize_av_separate_results":dict({"mixed_mixture":batch["mixed_mixture"],"mixtures":batch["mixtures"],
                                                                         "imgs":sep_input["imgs"],"frames":sep_input["frames"],"valid_nums":sep_input["valid_nums"]}
                                                                        ,**sep_output),
                                   **sep_output
                                   }
                self.net_wrapper.visualize(visualize_input)

            if self.opts.log_sir:
                signals=sep_output["est_components"].flatten(start_dim=1,end_dim=2)
                log_input["SIR"]=SIR_calculation(signals,signals).mean()
                log_input["SAR"] = SAR_calculation(sep_output["est_components"].sum(2)
                                                   , batch["mixtures"]
                                                   ).mean()

        self.pl_log(log_input)
        if str(self.batch_t) in self.opts.lambda_independence_incresing_step:
            self.opts.lambda_ind *= 10
        self.batch_t += 1
        return

    def on_validation_start(self) -> None:
        self.net_wrapper.av_separator.on_validation_start()

    def validation_step(self,batch,batchid,dataloader_idx=0) :
        if self.opts.volumn_normalize:
            volumn = torch.norm(batch["mixed_mixture"], dim=1)
            if "mean_volumn" in self.opts:
                volumn =volumn/self.opts.mean_volumn
            batch["mixed_mixture"]=batch["mixed_mixture"]/(volumn[:,None]+1e-8)
            batch["mixtures"] = batch["mixtures"]/(volumn[:,None,None]+1e-8)
        B,NumMix,Objects,_,_,_ = batch["objects"].size()
        sep_input = {"imgs": batch["objects"].flatten(start_dim=1, end_dim=2),"frames":batch["frames"],
                     "mixed_mixture": batch["mixed_mixture"],"valid_nums":batch["valid_nums"],"labels":batch["labels"].flatten(start_dim=1,end_dim=2)}
        sep_output = self.av_sep_forward(sep_input, Objects,self.net_wrapper.av_separator)


        if self.opts.volumn_normalize:
            sep_output["est_components"]=sep_output["est_components"]*(volumn[:,None,None,None])
            sep_output["est_mel_mag"]=sep_output["est_mel_mag"]*(volumn[:,None,None,None,None])
            batch["mixed_mixture"]=batch["mixed_mixture"]*(volumn[:,None])
            batch["mixtures"] = batch["mixtures"]*(volumn[:,None,None])
            sep_input["mixed_mixture"] = sep_input["mixed_mixture"] * (volumn[:, None])

        if self.global_rank==0 and batchid==0:
            visualize_input = {"vis_dir":join(self.opts.exp_vis_root,"Visualizations","batch%d"%self.batch_t),
                               "visualize_av_separate_results":dict({"mixtures":batch["mixtures"]},**sep_input,**sep_output),
                               **sep_output
                               }

            self.net_wrapper.visualize(visualize_input)

        validate_input=combine_dictionaries([{"mixtures":batch["mixtures"]},batch,sep_input,sep_output
                                                # ,critics_output
                                             ])

        bss_eval_output=self.net_wrapper.validate({"mixtures":batch["mixtures"],"mixed_mixture":batch["mixed_mixture"],**sep_output})#(B,NumMix,3)

        validate_input.update(bss_eval_output)
        log=self.net_wrapper.av_separator.validation_step(validate_input)

        ### log ###
        log.update({**bss_eval_output["metrics"]})

        self.pl_log(log,True)
        return

    def on_validation_end(self) -> None:
        #------- save best model -------
        metric=self.trainer.logged_metrics['hp/sdr']
        if self.best_metric<metric and self.global_rank==0:
            print("saving the best model")
            if not self.trainer.sanity_checking:
                # do something only for normal validation
                self.best_metric=metric
                self.trainer.save_checkpoint(join(self.opts.exp_vis_root,"sdr%3f_sir%3f_sar%3f_state_dict.ckpt"%(metric,self.trainer.logged_metrics['hp/sir'],self.trainer.logged_metrics['hp/sar'])))
        #-------------------------------
        self.net_wrapper.av_separator.on_validation_end()

    def input_volumn_normalize_wrap(self,sep_input):
        volumn = torch.norm(sep_input["mixed_mixture"], dim=1)
        if "mean_volumn" in self.opts:
            volumn = volumn / self.opts.mean_volumn
        if self.opts.volumn_normalize:
            sep_input["mixed_mixture"]=sep_input["mixed_mixture"]/(volumn[:,None]+1e-8)
        return sep_input,volumn

    def output_volumn_normalize_wrap(self,sep_output,volumn):
        if self.opts.volumn_normalize:
            sep_output["est_components"]=sep_output["est_components"]*(volumn[:,None,None,None])
            sep_output["est_mel_mag"]=sep_output["est_mel_mag"]*(volumn[:,None,None,None,None])
        return sep_output

    def test_separation(self,batch,vis_dir="",log_prefix=""):
        '''
        batch:
            # Video1: (B)
            # Ins1: (B)
            # Video2: (B)
            # Ins2: (B)
            mixed_audio: (B,Nsec)
            audios: (B,NumMix,Nsec)
            objects: (B,NumMix,MaxObject,3,H,W)
        '''

        mixed_audio,audios,objects=[batch["mixed_audio"],batch["audios"],batch["objects"]]
        B,AudLen=mixed_audio.size()
        _,NumMix,Objects,_,_,_=objects.size()
        _,mix_mel_mag,_=self.net_wrapper.stftMelModule.cal_mel_mag_from_wav(mixed_audio)

        sliding_window_start=0
        #slide on mixed_audio
        sum_sep_audio= torch.zeros_like(audios)
        overlap_count=torch.zeros_like(sum_sep_audio)
        while sliding_window_start + self.opts.samples_per_window < AudLen:
            st,ed=[sliding_window_start,sliding_window_start + self.opts.samples_per_window]

            mixture= mixed_audio[:,st:ed]
            sep_input = {"imgs": batch["objects"].flatten(start_dim=1, end_dim=2),"frames":batch["frames"],
                         "mixed_mixture": mixture,"valid_nums":batch["valid_nums"],"labels":batch["labels"].flatten(start_dim=1,end_dim=2)}
            sep_input,volumn=self.input_volumn_normalize_wrap(sep_input)
            sep_output = self.av_sep_forward(sep_input, Objects,self.test_separator)
            sep_output=self.output_volumn_normalize_wrap(sep_output,volumn)

            est_clip_audio=sep_output["est_components"].sum(2)
            sum_sep_audio[:,:,st:ed] +=est_clip_audio

            overlap_count [:,:,st:ed]+= 1
            sliding_window_start = sliding_window_start + int(self.opts.hop_size)

        # deal with the last segment
        mixture = mixed_audio[:,-self.opts.samples_per_window:]
        sep_input = {"imgs": batch["objects"].flatten(start_dim=1, end_dim=2),"frames":batch["frames"],
                     "mixed_mixture": mixture,"valid_nums":batch["valid_nums"],"labels":batch["labels"].flatten(start_dim=1,end_dim=2)}
        sep_input, volumn = self.input_volumn_normalize_wrap(sep_input)
        sep_output = self.av_sep_forward(sep_input, Objects,self.test_separator)
        sep_output = self.output_volumn_normalize_wrap(sep_output, volumn)

        est_clip_audio = sep_output["est_components"].sum(2)
        sum_sep_audio[:,:,-self.opts.samples_per_window:] +=est_clip_audio
        overlap_count[:,:,-self.opts.samples_per_window:] +=1

        #metric
        avged_sep_audio= sum_sep_audio/overlap_count

        #
        result,record_results = sep_val(audios, avged_sep_audio, B)
        sdr, sir, sar=result
        mixed_audio1=torch.stack([mixed_audio]*audios.size(1),dim=1)
        result_mix, _ = sep_val(audios, mixed_audio1, mixed_audio1.size(0))

        log_inputs = {log_prefix+"_test_sdr": sdr, log_prefix+"_test_sir": sir, log_prefix+"_test_sar": sar,
                      log_prefix+"_test_sdri": sdr-result_mix[0],log_prefix+"_test_siri": sir-result_mix[1]}
        self.pl_log(log_inputs, True)

        if vis_dir:
            est_mel_mask, _, est_mel_mag, _ = self.net_wrapper.gt_mel_mask(mixed_audio,avged_sep_audio)

            visualize_input = {"vis_dir":join(self.opts.exp_vis_root,"Visualizations","batch%d"%self.batch_t),
                               "visualize_av_separate_results": {"imgs":objects.flatten(start_dim=1,end_dim=2),"mixed_mixture":mixed_audio,
                                                                  "mixtures": audios,
                                                                  "est_components": avged_sep_audio.unsqueeze(2),
                                                                  "est_mel_mask":est_mel_mask.unsqueeze(2),"est_mel_mag":est_mel_mag.unsqueeze(2),
                                                                 "valid_nums":torch.ones(B,NumMix,Objects)}
                               ,**sep_output
                               }
            self.net_wrapper.visualize(visualize_input)
        return avged_sep_audio

    def on_test_start(self) -> None:
        pass

    def on_test_end(self) -> None:
        pass

    def test_step(self,batch,batchid,dataloader_idx=0):
        '''
        batch:
            # Video1: (B)
            # Ins1: (B)
            # Video2: (B)
            # Ins2: (B)
            mixed_audio: (B,Nsec)
            audios: (B,NumMix,Nsec)
            objects: (B,NumMix,MaxObject,3,H,W)
        '''
        if dataloader_idx==0:
            prefix="2-mix"
        elif dataloader_idx==1:
            prefix="3-mix"
        else:
            raise AttributeError("")

        mixed_audio, audios, objects = [batch["mixed_audio"], batch["audios"], batch["objects"]]
        _,NumMix,Objects,_,_,_=objects.size()

        self.test_separator=self.net_wrapper.av_separator
        self.test_critics=self.net_wrapper.critics

        if batchid % self.opts.n_visualize == 0 and self.opts.exp_vis_root and self.global_rank == 0:
            vis_dir = join(self.opts.exp_vis_root, "Visualizations",
                           "dataloader%d_batch%d" % (dataloader_idx, batchid))
        else:
            vis_dir = ""
        self.test_separation(batch, vis_dir,prefix)

    def load_av_separator(self, checkpoint) -> None:
        """Function that loads part of the model weights from a given checkpoint file.
        Note: If the checkpoint model architecture is different then `self`, only the common parts will be loaded.
        :param checkpoint: Path to the checkpoint containing the weights to be loaded.
        """
        print(f"loading part of model weights.")
        checkpoint = torch.load(checkpoint, map_location="cpu")
        pretrained_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "critics" not in k}
        # 2. update the model state with the pretrained weights
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        return self
