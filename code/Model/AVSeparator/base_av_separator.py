from torch import nn

class AVSeparator(nn.Module):
    def __init__(self,opts):
        super().__init__()

    def forward(self,inputs):
        '''
        -------Inputs------
        imgs:B,NumMix*Objects,3,H,W
        frames:B,NumMix,3,H,W
        mixed_mixture:  B,AudLen
        valid_num:(B,NumMix*Objects)
        -------------------

        -------Outputs------
        -SepResults
        est_components:B,NumMix*Objects,C,AudLen
        est_score_map,est_mel_mask,est_mel_mag:B,NumMix*Objects,C,F,T
        -LocResults
        cos_map:(B,NumMix*Objects,C,H,W)
        sound_localization:(B,NumMix*Objects,C,H,W)
        non_sounding_localization:(B,NumMix*Objects,1,H,W)
        -LeanedRepresentation
        fv:(B,NumMix*Objects,D,H,W)
        visual_related_srcs_fa_com:(B,NumMix*Objects,C,D)
        sp_att_fv_com:(B,NumMix*Objects,C,D)
        visual_related_srcs_att_fa:(B,NumMix*Objects,C,D,F,T)
        visual_related_srcs_att_fa_pooled:(B,NumMix*Objects,C,D)
        -------------------
        '''

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

        pass
