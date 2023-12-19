import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
# datatil提供了分割数据集的方法
from torch.utils.data import DataLoader,random_split
data_root="./practice/data"
from easydict import EasyDict
import yaml
from functools import partial
from .Datasets.SepValDataset import Det_Sep_Val_Dataset
from .Datasets.SyntheticDuetDataset import Synthetic_Duet_Train_Dataset as Det_Synthetic_Duet_Train_Dataset
from .Datasets.SoloDataset import Det_Solo_Train_Dataset
from .Datasets.TestDataset import Det_Sep_Test_Dataset
from torch.utils.data.dataloader import default_collate
import copy

test_path="./config/data/SyntheticDuet/test.yaml"
det_train_path="./config/data/SyntheticDuet/train.yaml"
det_solo_train_path="./config/data/SyntheticDuet/solo_train.yaml"
det_sep_val_config_path="./config/data/SyntheticDuet/sep_val.yaml"


def path_collate(batch):
    new_batch = []
    paths = []
    for _batch in batch:
        paths.append(_batch["paths"])
        _batch.pop("paths")
        new_batch.append(_batch)
    batch=default_collate(new_batch)
    batch.update({"paths":paths})
    return batch
def load_config(path):
    return EasyDict(yaml.full_load(open(path)))
class SyntheticDuetDataModule(pl.LightningDataModule):
    def __init__(self,opts):
        super().__init__()
        self.opts = opts

        self.sep_val_opt=load_config(det_sep_val_config_path)
        self.solo_train_opt=load_config(det_solo_train_path)
        self.train_opt = load_config(det_train_path)
        self.SoloTrainDataset= Det_Solo_Train_Dataset
        self.TrainDataset= Det_Synthetic_Duet_Train_Dataset
        self.ValDataset = Det_Sep_Val_Dataset

        self.SepValDataset=self.ValDataset

        if opts.train_on_3mix:
            self.solo_train_opt.num_mix=3
            self.sep_val_on_testset_opt.num_mix=3
            self.sep_val_opt.num_mix=3

        if self.opts.seen_heard_test:
            self.two_mix_test_opt = load_config("./config/data/SyntheticDuet/seen_heard_test.yaml")
        else:
            self.two_mix_test_opt = load_config(test_path)

        self.train_opt.use_RAM=opts.use_RAM
        self.solo_train_opt.use_RAM=opts.use_RAM
        self.two_mix_test_opt.cat_dets=self.opts.cat_dets
        self.three_mix_test_opt=copy.deepcopy(self.two_mix_test_opt)
        self.three_mix_test_opt.num_mix=3
        self.DataLoader = partial(DataLoader, num_workers=opts.num_workers, drop_last=True, shuffle=True, pin_memory=True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.opts.train_dataset == "SyntheticDuet":
            return self.DataLoader(self.TrainDataset(self.train_opt), self.opts.batch_size)
        elif self.opts.train_dataset == "MUSIC-Solo":
            return self.DataLoader(self.SoloTrainDataset(self.solo_train_opt), self.opts.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return [self.DataLoader(self.ValDataset(self.sep_val_opt), self.opts.batch_size)]

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.opts.test_3mix:
            return [
                self.DataLoader(Det_Sep_Test_Dataset(self.three_mix_test_opt), self.opts.batch_size,
                                collate_fn=path_collate)
            ]
        else:
            return [self.DataLoader(Det_Sep_Test_Dataset(self.two_mix_test_opt), self.opts.batch_size,
                                    collate_fn=path_collate)
                    ]
