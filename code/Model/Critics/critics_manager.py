from .models.LatePair import LatePairCriticsModel
from .GanTrainAlgorithm.GeometricGan import GeometricGAN
class CriticsManager():
    def __init__(self):
        super().__init__()
        pass

    def configCritics(self,opts):
        '''
        -------Inputs------
        opts:
          gan_train_algorithm_type(WGAN-gp,WGAN,SNGAN,MMD-GAN,Geometric-GAN-gp)
          critics_model_type(pairtime-early,pairtime-late,pair-distance)
          critics_model_opts
          gan_train_algotithm_opts
        -------------------

        -------Outputs------
        Critics
        -------------------
        '''
        # config critics models
        if opts.critics_model_type=="pairtime-late":
            critics_model = LatePairCriticsModel(opts.critics_model)
        else:
            raise AttributeError("Unknown Critics Model Type")

        # config GAN train algorithm for critics and G
        if opts.gan_train_algorithm_type=="GeoGAN":
            critics = GeometricGAN(critics_model, opts.gan_train_algorithm)
        else:
            raise AttributeError("Unknown Gan Train Algorithm Type")

        return critics