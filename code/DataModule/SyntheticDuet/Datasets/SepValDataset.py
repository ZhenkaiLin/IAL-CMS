
import random
import csv
import torch
from .BaseDataset import Base_Dataset

class Det_Sep_Val_Dataset(Base_Dataset):
    def __init__(self,  opt):
        super(Det_Sep_Val_Dataset, self).__init__(opt,"val")

        self.solo_sample_list=[]
        for row in csv.reader(open(opt.solo_csv, 'r'), delimiter=','):
            if len(row) < 2:
                continue
            self.solo_sample_list.append(row)
        print('#solo clip samples: {}'.format(len(self.solo_sample_list)))
        random.shuffle(self.solo_sample_list)
        # self.solo_sample_list=self.solo_sample_list[:300]
        self.cls2id={"acoustic_guitar":0,"clarinet":1,"saxophone":2,"violin":3,"flute":4,"cello":5,"trumpet":6,"tuba":7,
           "accordion":8,"xylophone":9,"erhu":10}

    def __len__(self):
        return len(self.solo_sample_list)

    def sample_N_different_instrument_clip(self,N):
        instruments={}
        infos=[]
        for i in range(N):
            idx = random.randint(0, self.__len__() - 1)
            wav_path,_,_=self.solo_sample_list[idx]
            ins=wav_path.split("/")[-2]
            while ins in instruments:
                idx = random.randint(0, self.__len__() - 1)
                wav_path, _, _ = self.solo_sample_list[idx]
                ins = wav_path.split("/")[-2]
            infos.append(self.solo_sample_list[idx])
            instruments[ins]=""
        return infos

    def sample_N_Solo(self,N):
        clips_infos=self.sample_N_different_instrument_clip(N)

        mixtures = []
        objects = []
        labels = []
        frames=[]
        pseudo_labels = []
        for i in range(N):
            mixture_objects = torch.zeros([self.max_object, 3, 224, 224])
            mixture_labels = torch.zeros([self.max_object])
            mixture_pseudo_labels = torch.zeros([self.max_object])
            clip_audios = []
            for j in range(1):
                wav_path, frame_dir, detection_path = clips_infos[i+j]
                ins = wav_path.split("/")[-2]
                clip_audios.append(self.load_audio(wav_path))
                clip_objects, valid_num, pseudo_label, _,frame= self.load_objects(detection_path, frame_dir,need_frame=True)
                assert valid_num == 1
                mixture_objects[j] = clip_objects[0]
                mixture_labels[j] = self.cls2id[ins]
                frames.append(frame)
                mixture_pseudo_labels[j] = pseudo_label[0]-1
            mixture = torch.stack(clip_audios).mean(dim=0)
            # ->mixture: mixture,mixture_objects,mixture_labels
            mixtures.append(mixture)
            objects.append(mixture_objects)
            labels.append(mixture_labels)
            pseudo_labels.append(mixture_pseudo_labels)
        mixtures = torch.stack(mixtures)  # (NumMix,AudLen)
        mixed_mixture = torch.sum(mixtures, dim=0)  # (AudLen)
        objects = torch.stack(objects)  # (NumMix,MaxObject,3,224,224)
        labels = torch.stack(labels) #(NumMix,MaxObject)
        frames=torch.stack(frames)
        pseudo_labels = torch.stack(pseudo_labels)  # (NumMix,Object
        ret_dict = {'mixed_mixture': mixed_mixture, 'mixtures': mixtures, 'objects': objects,
                    "labels": labels,"valid_nums":torch.ones_like(labels),"frames":frames,"pseudo_labels":pseudo_labels}
        return ret_dict
    def __getitem__(self, index):
        return self.sample_N_Solo(self.num_mix)

