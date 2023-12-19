from torch import nn

class BaseAlgorithm(nn.Module):
    def __init__(self,critics_model,opts):
        super().__init__()
        '''
        CriticsModel(pairtime - early, pairtime - late, pairtime - last)
        opts:
          GanTrainingType(WGAN-gp,WGAN,SNGAN,MMD-GAN,Geometric-GAN-gp)
        '''

        pass

    def critics_training_forward(self,inputs):
        '''
        -------Inputs------
        -SepResults
        sep:B,NumMix,3,H,W
        est_components:B,NumMic,C,AudLen
        est_score_map,est_mel_mask,est_mel_mag:B,NumMix,C,F,T
        opts:
          GanTrainingType(WGAN-gp,WGAN,SNGAN,MMD-GAN,Geometric-GAN-gp)
        -------------------

        -------Outputs------
        c_loss:
        est_dist_d:
        -------------------
        '''

        pass

    def generator_training_forward(self,inputs):
        '''
        -------Inputs------
        -SepResults
        sep:B,NumMix,3,H,W
        est_components:B,NumMic,C,AudLen
        est_score_map,est_mel_mask,est_mel_mag:B,NumMix,C,F,T
        opts:
          GanTrainingType(WGAN-gp,WGAN,SNGAN,MMD-GAN,Geometric-GAN-gp)
        -------------------

        -------Outputs------
        g_loss:
        est_dist_d:
        -------------------
        '''

        pass
