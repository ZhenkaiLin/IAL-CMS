import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from utils.fixedTqdm import LitProgressBar
from DataModule.SyntheticDuet.LightiningDataModule import SyntheticDuetDataModule

#模型和测试没准备
from LightningModule.AdversarialTrainLightingModule import AdversarialTrainLM
from options.options import Options
import sys
from os.path import join
def select_datamodule(opts):
    if opts.train_dataset == "SyntheticDuet":
        return SyntheticDuetDataModule(opts)
    elif opts.train_dataset == "MUSIC-Solo":
        return SyntheticDuetDataModule(opts)
    else:
        raise AttributeError("Unknown Datamodule")

if __name__ == '__main__':
    opts = Options().parse()
    sys.stderr=open("error"+opts.exp_name+".txt","a")
    # ----------------------
    # Load Lightning Module
    # ----------------------
    lm_class=AdversarialTrainLM
    if opts.checkpoint:
        # lm=load_checkpoint_ignore_size_mismatch(opts.checkpoint,lm_class=lm_class,map_location="cpu",opts=opts)
        if opts.only_load_separator:
            lm = lm_class(opts).load_av_separator(opts.checkpoint)
        else:
            lm = lm_class.load_from_checkpoint(opts.checkpoint, map_location="cpu", opts=opts, strict=False)
        print("Load Checkpoint Succesfully.")
    else:
        lm = lm_class(opts)

    # ----------------------
    # Load Data Module
    # ----------------------
    dm=select_datamodule(opts)

    # ----------------------
    # Load Trainer
    # ----------------------
    logger = TensorBoardLogger(save_dir=opts.exp_vis_root,name="log")
    bar=LitProgressBar()
    trainer = Trainer(gpus=opts.gpus, max_epochs=-1, logger=logger, log_every_n_steps=10,
                      num_sanity_val_steps=opts.num_sanity_val_steps,check_val_every_n_epoch=1,strategy="ddp")

    # ----------------------
    # Train
    # ----------------------
    if opts.test:
        sys.stdout = open(join(opts.exp_vis_root,"test_result.txt"), "a")
        trainer.test(lm,dm)
    elif opts.validate:
        # sys.stdout = open(join(opts.exp_vis_root,"validate_result.txt"), "a")
        trainer.validate(lm,dm)
    else:
        trainer.fit(lm,dm)