import random
import csv
import torch
from .BaseDataset import Base_Dataset
import numpy as np
import librosa
import pickle
import os
from PIL import Image
class Synthetic_Duet_Train_Dataset(Base_Dataset):
    def __init__(self, opt, split='train'):
        super(Synthetic_Duet_Train_Dataset, self).__init__(opt,split)
        self.opt=opt
        self.solo_sample_list=[]
        for row in csv.reader(open(opt.solo_csv, 'r'), delimiter=','):
            if len(row) < 2:
                continue
            self.solo_sample_list.append(row)
        print('#solo clip samples: {}'.format(len(self.solo_sample_list)))
        # self.solo_sample_list=self.solo_sample_list[:10]
        # self.detections=[]
        # self.wavs = []
        # self.clip_frames_list=[]
        # for wav_path,frame_dir,detection_path in self.solo_sample_list:
        #     detection=np.load(detection_path)
        #     random.shuffle(detection)
        #     detection=detection[:8]
        #     self.detections.append(detection)
        #     wav, _ = librosa.load(wav_path, sr=self.audRate)
        #     self.wavs.append(wav)
        #     frame_list=[]
        #     for detect in detection:
        #         frame_id = int(detect[0])
        #         frame_path = os.path.join(frame_dir, "%06d.png" % frame_id)
        #         # frame = Image.open(frame_path)
        #         frame=self.totensor(Image.open(frame_path).convert('RGB'))
        #         frame_list.append(frame)
        #     self.clip_frames_list.append(frame_list)
        #
        # pickle.dump(self.detections,open("/data5/yan/lzk/MUSICDetection/train_detections.pkl","wb"))
        # pickle.dump(self.wavs,open("/data5/yan/lzk/MUSIC/train_wavs.pkl", "wb"))
        # pickle.dump(self.clip_frames_list,open("/data5/yan/lzk/MUSIC/train_frames_tensor.pkl", "wb"))
        if opt.use_RAM:
            self.detections=pickle.load(open("DataModule/SyntheticDuet/indicators/train_detections.pkl","rb"))
            self.wavs = pickle.load(open("DataModule/SyntheticDuet/indicators/train_wavs.pkl","rb"))
            self.clip_frames_list = pickle.load(open("DataModule/SyntheticDuet/indicators/train_frames.pkl", "rb"))

            self.data_len=len(self.solo_sample_list)
            if self.split == 'train':
                c = list(zip(self.solo_sample_list,self.detections,self.wavs,self.clip_frames_list))
                random.shuffle(c)
                self.solo_sample_list,self.detections,self.wavs,self.clip_frames_list=zip(*c)
                self.solo_sample_list = self.solo_sample_list * opt.dup_times
            assert (self.wavs[0]==librosa.load(self.solo_sample_list[0][0], sr=self.audRate)[0]).all()
        else:
            self.detections=pickle.load(open("DataModule/SyntheticDuet/indicators/train_detections.pkl","rb"))
            self.wavs = pickle.load(open("DataModule/SyntheticDuet/indicators/train_wavs.pkl","rb"))
            self.data_len=len(self.solo_sample_list)
            if self.split == 'train':
                c = list(zip(self.solo_sample_list,self.detections,self.wavs))
                random.shuffle(c)
                self.solo_sample_list,self.detections,self.wavs=zip(*c)
                self.solo_sample_list = self.solo_sample_list * opt.dup_times
        self.cls2id={"acoustic_guitar":0,"clarinet":1,"saxophone":2,"violin":3,"flute":4,"cello":5,"trumpet":6,"tuba":7,
           "accordion":8,"xylophone":9,"erhu":10}

    def __len__(self):
        return len(self.solo_sample_list)

    def sample_N_different_instrument_clip(self,N):
        instruments={}
        infos=[]
        idxs=[]
        for i in range(N):
            idx = random.randint(0, self.__len__() - 1)
            wav_path,_,_=self.solo_sample_list[idx]
            ins=wav_path.split("/")[-2]
            while ins in instruments:
                idx = random.randint(0, self.__len__() - 1)
                wav_path, _, _ = self.solo_sample_list[idx]
                ins = wav_path.split("/")[-2]
            infos.append(self.solo_sample_list[idx])
            idxs.append(idx)
            instruments[ins]=""
        return infos,idxs

    def sample_N_synthetic_duet(self,N):
        clips_infos,idxs=self.sample_N_different_instrument_clip(2*N)

        mixtures = []
        objects = []
        labels = []
        syn_frames=[]
        pseudo_labels=[]
        for i in range(N):
            mixture_objects = torch.zeros([self.max_object, 3, 224, 224])
            mixture_labels = torch.zeros([self.max_object])
            mixture_pseudo_labels= torch.zeros([self.max_object])
            clip_audios = []
            frames=[]
            for j in range(2):
                wav_path, frame_dir, detection_path = clips_infos[i*2+j]
                if self.opt.use_RAM:
                    detection = self.detections[idxs[i * 2 + j] % self.data_len]
                    wav=self.wavs[idxs[i * 2 + j] % self.data_len]
                    frame_list=self.clip_frames_list[idxs[i * 2 + j] % self.data_len]
                    # detection = None
                    # wav=None
                    ins = wav_path.split("/")[-2]
                    clip_audios.append(self.load_audio(wav_path,wav))
                    clip_objects, valid_num, pseudo_label, _,frame= self.load_objects(detection_path, frame_dir,detection,frame_list)
                else:
                    detection = self.detections[idxs[i * 2 + j] % self.data_len]
                    wav=self.wavs[idxs[i * 2 + j] % self.data_len]
                    ins = wav_path.split("/")[-2]
                    clip_audios.append(self.load_audio(wav_path,wav))
                    clip_objects, valid_num, pseudo_label, _,frame= self.load_objects(detection_path, frame_dir,detection)
                assert valid_num == 1
                mixture_objects[j] = clip_objects[0]
                mixture_labels[j] = self.cls2id[ins]
                mixture_pseudo_labels[j]=pseudo_label[0]-1
                frames.append(clip_objects[0])
            mixture = torch.stack(clip_audios).mean(dim=0)
            syn_frame=torch.cat(frames,dim=2)
            # ->mixture: mixture,mixture_objects,mixture_labels
            mixtures.append(mixture)
            objects.append(mixture_objects)
            labels.append(mixture_labels)
            pseudo_labels.append(mixture_pseudo_labels)
            syn_frames.append(syn_frame)
        mixtures = torch.stack(mixtures)  # (NumMix,AudLen)
        mixed_mixture = torch.sum(mixtures, dim=0)  # (AudLen)
        objects = torch.stack(objects)  # (NumMix,MaxObject,3,224,224)
        labels = torch.stack(labels) #(NumMix,Object)
        syn_frames=torch.stack(syn_frames)# (NumMix,MaxObject,3,224,448)
        pseudo_labels=torch.stack(pseudo_labels)#(NumMix,Object)
        ret_dict = {'mixed_mixture': mixed_mixture, 'mixtures': mixtures, 'objects': objects,
                    "labels": labels,"valid_nums":torch.ones_like(labels),"frames": syn_frames,
                    "pseudo_labels":pseudo_labels}
        return ret_dict
    def __getitem__(self, index):
        return self.sample_N_synthetic_duet(self.num_mix)