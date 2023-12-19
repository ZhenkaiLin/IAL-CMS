import argparse
import os
import torch
from easydict import EasyDict
import yaml
import sys
import shutil
import pytorch_lightning as pl
import torch
def load_config(config_path):
    return EasyDict(yaml.full_load(open(config_path)))

def print_dict(d, indent=0,file=sys.stdout):
    for key, value in d.items():
        if isinstance(value, dict):
            print('\t' * indent + str(key),file=file)
            print_dict(value, indent + 1,file=file)
        else:
            print('\t' * (indent) + str(key),":",str(value),file=file)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized=False
    def initialize(self):
        ### algorithm###
        self.parser.add_argument("--algorithm", type=str, default="")
        self.parser.add_argument("--seen_heard_test", type=str, default="")
        self.parser.add_argument("--sr", type=int, default=11025)
        self.parser.add_argument("--log_sir",type=str,default="1")
        self.parser.add_argument("--use_RAM", type=str, default="")

        ### model ###
        self.parser.add_argument("--lmconfig", type=str)
        self.parser.add_argument("--checkpoint", default='', type=str)
        self.parser.add_argument("--only_load_separator", default='', type=str)
        self.parser.add_argument("--lambda_independence_incresing_step",default=[], nargs="*")
        self.parser.add_argument("--lambda_avc_incresing_step", default=[], nargs="*")

        ### dataset/dataloader ###
        self.parser.add_argument("--train_dataset", type=str, default="SyntheticDuet")
        self.parser.add_argument("--batch_size", type=int, default=15)
        self.parser.add_argument("--num_workers", type=int, default=12)
        self.parser.add_argument("--det_num", type=str, default="det_one")
        self.parser.add_argument("--train_on_3mix", default="", type=str)

        ### trainer ###
        self.parser.add_argument("--gpus", type=int, nargs="*")
        self.parser.add_argument("--num_sanity_val_steps", type=int, default=1)
        self.parser.add_argument("--max_epochs", type=int, default=100)

        ### log ###
        self.parser.add_argument("--vis_root", type=str)
        self.parser.add_argument("--n_visualize", type=int, default=400, help="interval betwen visualization")

        ### mode ###
        self.parser.add_argument("--test", default='', type=str)

        ### test ###
        self.parser.add_argument("--samples_per_window", default=65535, type=int)
        self.parser.add_argument("--hop_size", default=2205, type=int)
        self.parser.add_argument("--test_3mix", default="", type=str)

        ### analyze the statics of separation results ###
        self.parser.add_argument("--validate", default="", type=str)
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = EasyDict(vars(self.parser.parse_args()))
        self.opt.update(load_config(self.opt.lmconfig))
        self.opt.exp_vis_root=os.path.join(self.opt.vis_root,self.opt.exp_name)
        if os.path.exists(self.opt.exp_vis_root):
            print("Warning: Experiments Visualization Directory Already Exists")
            print("Warning: Experiments Visualization Directory Already Exists")
            print("Warning: Experiments Visualization Directory Already Exists")
        self.opt.cuda_visible_device = os.environ["CUDA_VISIBLE_DEVICES"]
        while os.path.exists(self.opt.exp_vis_root):
            self.opt.exp_vis_root = self.opt.exp_vis_root + "x"
        self.opt.lambda_ind /= 10 ** len(self.opt.lambda_independence_incresing_step)
        if self.opt.algorithm=="IAL-CMS":
            # preprocess options including exp_vis_root lambda_ind
            sopt=self.opt.model.av_separator
            lopt=sopt.loss
            lopt.lambda_avc /= 10 ** len(self.opt.lambda_avc_incresing_step)
            sopt.visual_net.dropout=lopt.avc_contrastive.dropout
            lopt.lambda_avc_incresing_step=self.opt.lambda_avc_incresing_step
            multi_scale_avc=self.opt.model.av_separator.multi_scale_avc
            sopt.av_module.multi_scale_avc=multi_scale_avc
            sopt.separate_net.multi_scale_avc=multi_scale_avc
            assert self.opt.train_generator_interval==1 or self.opt.train_critics_interval==1

        print('------------ Options -------------')
        print_dict(self.opt)
        print('-------------- End ----------------')

        # save to the disk
        os.makedirs(self.opt.exp_vis_root,exist_ok=True)
        opt_file_path=os.path.join(self.opt.exp_vis_root, 'opt.txt')
        with open(opt_file_path, 'w') as opt_file:
            print_dict(self.opt,file=opt_file)
        return self.opt



