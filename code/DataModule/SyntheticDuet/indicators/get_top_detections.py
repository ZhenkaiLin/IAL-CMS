import cv2
import numpy as np
import os
from os.path import join
import matplotlib.pyplot as plt
from PIL import Image
import random
det_root="dataroot/music_dataset/solo/solo_detect/tuba"
K=1
video_vis_frame_count={}

class2cid={"acoustic_guitar":4,"clarinet":7,"saxophone":9,"violin":12,"flute":13,"cello":2,"trumpet":11,"tuba":15,
           "accordion":14,"xylophone":8}

save_detsdir_root="dataroot/music_dataset/solo/solo_detect/"
for dir,subdir,files in os.walk(det_root):
    random.shuffle(files)
    for file in files:
        if file[-4:]==".npy":
            clip_id=file[:-4]
            video_cls_name=dir.split("/")[-1]
            video_id="_".join(file.split("_")[:-1])
            det_path=join(dir,file)
            frame_dir_path=det_path[:-4]
            save_detsdir_path=join(save_detsdir_root,video_cls_name)
            os.makedirs(save_detsdir_path,exist_ok=True)
            dets = np.load(det_path)
            det_dir={}
            for det in dets:
                frame_id = int(det[0])
                if not frame_id in det_dir:
                    det_dir[frame_id]=[det]
                else:
                    det_dir[frame_id].append(det)
            clip_dets=[]
            for frame_id,f_dets in det_dir.items():
                cls_dets={}
                for det in f_dets:
                    cls_id=int(det[1])
                    if video_cls_name=="erhu":
                        if video_id in ["DlqCn_xrNRU","37HdHAzJrOQ","-e5DuAUwBgA","-7l7M6vMm1k","0oKi3ARn640","8rb-thew50c","JEDgAVcR_yE"]:
                            continue
                        if det[2]> 0.94:
                            continue
                    elif video_cls_name=="xylophone":
                        if video_id in ["5lm9laLSORc","Jtq-HDS3Zdc","oQJVBH6ST7o","Sw8346DYwME"]:
                            continue
                        if not round(det[1]) ==class2cid[video_cls_name]:
                            continue
                    else:
                        if det[2]<0.9:
                            continue
                        if not round(det[1]) == class2cid[video_cls_name]:
                            continue
                    if not cls_id in cls_dets:
                        cls_dets[cls_id] = [det]
                    else:
                        cls_dets[cls_id].append(det)
                cls_top_dets=[]
                for cls_id,dets in cls_dets.items():
                    cls_top_dets.append(dets[np.stack(dets)[:,2].argmax()])
                if len(cls_top_dets)>=K:
                    top_dets=np.stack(cls_top_dets)[np.argsort(np.stack(cls_top_dets)[:,2])][-K:]
                else:
                    # print("less than one cls dets")
                    continue
                frame_path = os.path.join(frame_dir_path, "%06d.png" % frame_id)
                clip_dets.append(top_dets)

                # frame=cv2.imread(frame_path)
                # for det in f_dets:
                #     lu=det[3:5].astype(int)
                #     rb=det[5:7].astype(int)
                #     cv2.rectangle(frame, lu, rb, (255, 0, 0), 2)
                #     font = cv2.FONT_HERSHEY_SIMPLEX
                #     cv2.putText(frame, '{} {:.3f}'.format(det[1], det[2]),lu, font, 0.5, (0, 255, 255), 1)
                # # 图像，文字内容，坐标(右上角坐标) ，字体，大小，颜色，字体厚度
                # cv2.imwrite(join(save_visdir_path,"%06d.png" % frame_id), frame)

                if video_id not in video_vis_frame_count:
                    video_vis_frame_count[video_id]=0
                if video_vis_frame_count[video_id]<200:
                    save_visdir_path = join(save_detsdir_path, clip_id)
                    os.makedirs(save_visdir_path, exist_ok=True)
                    video_vis_frame_count[video_id]+=1
                    frame=cv2.imread(frame_path)
                    for det in top_dets:
                        lu=det[3:5].astype(int)
                        rb=det[5:7].astype(int)
                        cv2.rectangle(frame, lu, rb, (255, 0, 0), 2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, '{} {:.3f}'.format(det[1], det[2]),lu, font, 0.5, (0, 255, 255), 1)
                    # 图像，文字内容，坐标(右上角坐标) ，字体，大小，颜色，字体厚度
                    cv2.imwrite(join(save_visdir_path,"%06d.png" % frame_id), frame)

            if len(clip_dets) >0:
                clip_dets=np.concatenate(clip_dets,axis=0)
                np.save(join(save_detsdir_path,file),clip_dets)

