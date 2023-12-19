
import random
import csv
import torch
from .BaseDataset import Base_Dataset
from itertools import combinations
import numpy as np
def get_ins_name(info):
    wav_path=info[0]
    ins = wav_path.split("/")[-2]
    return ins

class Det_Sep_Test_Dataset(Base_Dataset):
    def __init__(self,  opt):
        super(Det_Sep_Test_Dataset, self).__init__(opt,"test")

        self.opt = opt
        random.seed(opt.seed)

        self.solo_sample_list=[]
        for row in csv.reader(open(opt.solo_csv, 'r'), delimiter=','):
            if len(row) < 2:
                continue
            self.solo_sample_list.append(row)
        print('#test clip samples: {}'.format(len(self.solo_sample_list)))

        if opt.train_dataset == "MUSICDeut":
            instruments = {'acoustic_guitar': [], 'clarinet': [], 'saxophone': [], 'violin': []}
            ins = list(instruments.keys())
            assert opt.num_mix==2
            test_pairs = [('acoustic_guitar', 'clarinet'),
                          ('acoustic_guitar', 'saxophone'),
                          ('acoustic_guitar', 'violin')]  # 这里的2实际为混合的音频个数
        elif opt.train_dataset == "SyntheticDuet" or opt.train_dataset == "MUSICSolo":
            instruments = {'accordion': [], 'acoustic_guitar': [], 'cello': [],
                           'clarinet': [], 'erhu': [], 'flute': [], 'saxophone': [],
                           'trumpet': [], 'tuba': [], 'violin': [], 'xylophone': []}
            ins = list(instruments.keys())
            test_pairs = list(combinations(ins, int(opt.num_mix)))  # 这里的2实际为混合的音频个数

        else:
            instruments = {'accordion': [], 'acoustic_guitar': [], 'bagpipe': [], 'banjo': [],
                           'bassoon': [], 'cello': [], 'clarinet': [], 'congas': [], 'drum': [],
                           'electric_bass': [], 'erhu': [], 'flute': [], 'guzheng': [],
                           'piano': [], 'pipa': [], 'saxophone': [], 'trumpet': [], 'tuba': [],
                           'ukulele': [], 'violin': [], 'xylophone': []}
            ins = list(instruments.keys())
            test_pairs = list(combinations(ins, int(opt.num_mix)))  # 这里的2实际为混合的音频个数


        for info in self.solo_sample_list:
            ins = get_ins_name(info)
            instruments[ins].append(info)

        self.test_clip_pair_list=[]
        for pair in test_pairs:
            for iters in range(opt.sample_times_per_video_pair):
                test_clip_pair=[]
                for ins in pair:
                    test_clip_pair.append([ins, random.choice(instruments[ins])])
                self.test_clip_pair_list.append(test_clip_pair)
        random.shuffle(self.test_clip_pair_list)
        print('#test clip pair: {}'.format(len(self.test_clip_pair_list)))
        self.cls2id={"acoustic_guitar":0,"clarinet":1,"saxophone":2,"violin":3,"flute":4,"cello":5,"trumpet":6,"tuba":7,
           "accordion":8,"xylophone":9,"erhu":10}

    def __len__(self):
        return len(self.test_clip_pair_list)

    def __getitem__(self, index):
        '''
            # Video1:
            # Ins1:
            # Video2:
            # Ins2:
            mixed_audio: (10sec)
            audios: (2,10sec)
            objects: (2,1,3,H,W)
            labels: (2,1) (NumMix,MaxObject)
        '''
        clips_infos=[ clip[1] for clip in self.test_clip_pair_list[index]]
        paths=[clip[2] for clip in clips_infos]
        mixtures = []
        objects = []
        labels = []
        frames=[]
        pseudo_labels = []
        for i in range(self.num_mix):
            mixture_objects = torch.zeros([self.max_object, 3, 224, 224])
            mixture_labels = torch.zeros([self.max_object])
            mixture_pseudo_labels = torch.zeros([self.max_object])
            clip_audios = []
            for j in range(1):
                wav_path, frame_dir, detection_path = clips_infos[i+j]
                ins = wav_path.split("/")[-2]
                clip_audios.append(self.load_audio(wav_path))
                clip_objects, valid_num,  pseudo_label, _,frame= self.load_objects(detection_path, frame_dir,need_frame=True)
                assert valid_num == 1
                mixture_objects[j] = clip_objects[0]
                mixture_labels[j] = self.cls2id[ins]
                mixture_pseudo_labels[j] = pseudo_label[0]-1
                if self.opt.cat_dets:
                    frames.append(clip_objects[0])
                else:
                    frames.append(frame)
            mixture = torch.stack(clip_audios).mean(dim=0)
            # ->mixture: mixture,mixture_objects,mixture_labels
            mixtures.append(mixture)
            objects.append(mixture_objects)
            labels.append(mixture_labels)
            pseudo_labels.append(mixture_pseudo_labels)
        mixtures = torch.stack(mixtures)  # (NumMix,AudLen)
        mixed_mixture = torch.sum(mixtures, dim=0)  # (AudLen)
        objects = torch.stack(objects)  # (NumMix,MaxObject,3,224,224)
        labels = torch.stack(labels)
        frames=torch.stack(frames)
        pseudo_labels = torch.stack(pseudo_labels)  # (NumMix,Object)
        ret_dict = {'mixed_audio': mixed_mixture, 'audios': mixtures, 'objects': objects,
                    "labels": labels,"valid_nums":torch.ones_like(labels),"paths":paths,"frames":frames,
                    "pseudo_labels":pseudo_labels}
        return ret_dict

