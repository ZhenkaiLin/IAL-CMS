import librosa
import random
import csv
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
from torchvision.transforms import functional as tr_functional
from PIL import Image
import os
import pickle

class Base_Dataset(torchdata.Dataset):
    def __init__(self, opt, split='train'):
        super(Base_Dataset, self).__init__()
        # params
        self.opt=opt
        self.max_object=opt.max_object
        self.num_mix=opt.num_mix
        self.audRate = opt.sr
        self.audLen=opt.audLen
        self.audSec = 1. * self.audLen / self.audRate

        self.split = split
        self.seed = opt.seed
        random.seed(self.seed)
        # initialize transform
        self._init_transform()

    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transforms_list=[]

        transforms_list.append(transforms.Resize([256,256]))
        if self.split == 'train':
            transforms_list.extend([
                transforms.RandomCrop([224,224]),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(mean, std)])
        else:
            transforms_list.extend([
                transforms.CenterCrop([224,224]),
                transforms.Normalize(mean, std)])
        self.object_transform=transforms.Compose(transforms_list)

        transforms_list = []
        transforms_list.append(transforms.Resize([256, 256]))
        if self.split == 'train':
            transforms_list.extend([
                transforms.RandomCrop([224, 224]),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            transforms_list.extend([
                transforms.CenterCrop([224, 224]),
                # transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        self.img_transform=transforms.Compose(transforms_list)
        self.totensor=transforms.ToTensor()

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        img=self.img_transform(img)
        return img

    def load_audio(self,wav_path,wav=None):
        if wav is None:
            wav, _ = librosa.load(wav_path, sr=self.audRate)
        #zero padding if wav length is less than AudLen
        if len(wav) < self.audLen:
            pad_wav=np.zeros(self.audLen,dtype=np.float32)
            pad_wav[:len(wav)]=wav
            wav=pad_wav
        #sample win lenth
        st=random.randint(0, len(wav)-self.audLen)
        audio=wav[st:st+self.audLen]
        #augment

        if self.split == 'train':
            audio = audio * (0.5 + random.random())

        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.
        return torch.from_numpy(audio)

    def load_objects(self,detection_path,frame_dir,detections=None,frame_list=None,need_frame=False):
        objects=torch.zeros([self.max_object,3,224,224])
        pseudo_labels =torch.zeros([self.max_object])
        if detections is None:
            detections=np.load(detection_path)

        idx=random.randint(0,len(detections)-1)
        detection=detections[idx]

        if frame_list is None:
            frame_id=int(detection[0])
            frame_path=os.path.join(frame_dir,"%06d.png"%frame_id)

            frame=self.totensor(Image.open(frame_path).convert('RGB'))
        else:
            frame=self.totensor(frame_list[idx].convert('RGB'))

        object=frame[:,int(detection[-3]):int(detection[-1]),int(detection[-4]):int(detection[-2])]
        pseudo_labels[0]=int(detection[1])
        object=self.object_transform(object)

        if need_frame:
            frame=self.img_transform(frame)
        objects[0]=object
        return objects,1,pseudo_labels,None,frame