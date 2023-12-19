import os
import h5py
import numpy as np
import random
indicator_path="./DataModule/SyntheticDuet/indicators"
fp_train=h5py.File("/data/lzk/dataset/music_dataset/solo_train.h5","w")
fp_val=h5py.File("/data/lzk/dataset/music_dataset/solo_val.h5","w")
infos=[]
test_infos=[]
test_video=["_oUTOskOwXs","I5IYfx7ontI","YR6kjiPyNHU","iB17xqmFw3A","AAUm9njcMRk","U2NMdF5yzdE","x5Bzb6R73JQ","-tY40Ev8IzE","91MhpKVea30","XdoR3Mw217M","IZx6ZXWuOGY"]
for root,dir,files in os.walk("/data/lzk/dataset/music_dataset/solo/solo_detect/"):
    for file in files:
        if file[-4:]==".npy":
            video_id="_".join(file[:-4].split("_")[:-1])
            detection_path=os.path.join(root,file)
            frame_dir=detection_path.replace("lzk","mashuo").replace("solo_detect","solo_extract")[:-4]
            wav_path=frame_dir.replace("solo_extract","solo_audio_resample")+".wav"
            if video_id in test_video:
                test_infos.append([wav_path, frame_dir, detection_path])
            else:
                infos.append([wav_path, frame_dir, detection_path])


random.shuffle(infos)
train_num=int(len(infos)*0.8)

train_infos=infos[:train_num]
val_infos=infos[train_num:]

def save_infos(infos,name,indicator_path):
    filename = '{}.csv'.format(os.path.join(indicator_path, name))
    with open(filename, 'w') as f:
        for item in infos:
            f.write(",".join(item) + '\n')
    print('{} items saved to {}.'.format(len(infos), filename))

save_infos(train_infos,"train_solo_clip",indicator_path)
save_infos(val_infos,"val_solo_clip",indicator_path)
save_infos(test_infos,"test_solo_clip",indicator_path)