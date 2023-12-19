from .base_algorithm import BaseAlgorithm
from torch import nn
from torch import autograd
import random
import torch
from utils.utils import random_sample_from_est_mel_mag,random_sample_components_from_bank
class GeometricGAN(BaseAlgorithm):
    def __init__(self,critics_model,opts):
        super().__init__(critics_model,opts)
        '''
        gan_train_algotithm_opts:
            C
        '''
        self.critics_model=critics_model
        self.opts=opts


    def critics_training_forward(self,inputs):
        '''
        --------- inputs ---------
        -SepResults
            est_components:B,NumMix,Objects*C,AudLen
            est_score_map,est_mel_mask,est_mel_mag:B,NumMix,Objects*C,F,T
        --------------------------

        --------- outputs ---------
        **critics_outputs
        c_loss
        est_dist_d
        -GradientOutputs
            gradients_penalty:R
            gradients_norm:R
        --------------------------
        '''
        critics_model_outputs=self.critics_model(inputs)
        independent_pair_score=critics_model_outputs["independent_pair_score"]
        dependent_pair_score=critics_model_outputs["dependent_pair_score"]

        # Adversarial loss

        est_dist_d = torch.mean(independent_pair_score) - torch.mean(dependent_pair_score)
        c_loss = (self.critics_model.w.weight**2).sum()+self.opts.C*\
                 (torch.clip(1-independent_pair_score,min=0).mean()+
                  torch.clip(1+dependent_pair_score,min=0).mean())
                 #+self.opts.lambda_gp*gradients_penalty

        return dict({"c_loss":c_loss,"est_dist_d":est_dist_d},**critics_model_outputs
                    )

    def generator_training_forward(self,inputs):

        critics_model_outputs=self.critics_model(inputs)
        dependent_pair_score=critics_model_outputs["dependent_pair_score"]
        independent_pair_score = critics_model_outputs["independent_pair_score"]
        # modify from g_loss=-fake_validity as fake and true distributions are both the output of G
        est_dist_d = torch.mean(independent_pair_score) - torch.mean(dependent_pair_score)
        g_loss = est_dist_d

        est_mel_mag=inputs["est_mel_mag"]
        return dict({"gradients_norm":torch.zeros_like(est_mel_mag.mean()),"gradients":torch.zeros_like(est_mel_mag),"g_loss":g_loss,"est_dist_d":est_dist_d},**critics_model_outputs)

