from torch import nn

class CriticsBaseModel(nn.Module):
    def __init__(self,opts):
        super().__init__()
        '''
        Use the critics formulation of Geometric-GAN 
        Critics=Activation(<w,f(x)>)
        CriticsBaseModel corresponds to f
        x is a components pair sampled independently or dependently
        opts:
          critics_model_opts:
            input_f_dim
            input_t_dim
        '''


        pass

    def forward(self,inputs):
        '''
        -------Inputs------
        -SepResults
        est_components:B,NumMic,C,AudLen
        est_score_map,est_mel_mask,est_mel_mag:B,NumMix,C,F,T
        -------------------

        -------Outputs------
        independent_pair_feature:N,D
        dependent_pair_feature: M,D
        independent_pair_score:N
        dependent_pair_score: M
        -------------------
        '''

        pass