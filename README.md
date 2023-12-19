# Independency-Adversarial-Learning-for-Cross-modal-Sound-Separation
The codebase is under construction, but all the core codes are already here.
### Data Preparation
We use the public PyTorch implementation of Faster R-CNN (https://github.com/jwyang/faster-rcnn.pytorch) to train an object detector with a ResNet-101 backbone. The object detector is trained on ∼30k images of 15 object categories from the Open Images dataset. The 15 object categories include: Banjo, Cello, Drum, Guitar, Harp, Harmonica, Oboe, Piano, Saxophone, Trombone, Trumpet, Violin, Flute, Accordion, and Horn. The pre-trained detector is shared at Google Drive. Please refer to https://github.com/jwyang/faster-rcnn.pytorch for instructions on how to use the pre-trained object detector or train a new detector on categories of your interest. Use the pretrained-detector to generate object detections for both training and testing set, and save the object detection results of each video as one .npy file under /your_data_root/detection_results/.

The training data we use are 10s video clips. 80 frames are extracted at 8 fps from each 10s clip. Each .npy file contains the pooled object detections from these 10 frames. Audio segment is randomly cropped from the corresponding 10s clip during training. Each .npy file should contain all the object detections for that video video with each detection represented by 7 numbers (frame_index,class_id,confidence_score,four bounding box coordinates). See getDetectionResults.py for examples on how to obtain the .npy file. See Supp. for how we reduce the noise of the obtained detections. A script (getTopDetections.py) is provided for object detection data cleaning.

### Dataset Preparation
Download MUSIC dataset from: [github.com/roudimit/MUSIC_dataset](https://github.com/roudimit/MUSIC_dataset).
The training data we use are 10s video clips with audiorate 11025Hz. 80 frames are extracted at 8 fps from each 10s clip. Following [Co-Separation](https://github.com/rhgao/co-separation/tree/master), we use the public PyTorch implementation of Faster R-CNN (https://github.com/jwyang/faster-rcnn.pytorch) to generate object detections for both training and testing set, and save the object detection results of each video as one .npy file under /your_data_root/detection_results/. Each .npy file contains the pooled object detections from these 10 frames. We reduce the noise of the obtained detections by get_top_detections.py.
Then make training/validation index files by running:
```
python DataModule/SyntheticDuet/indicators/prepare_indicator.py
```
It will create index files train_solo_clip.csv/val_solo_clip.csv/test_solo_clip.csv with the following format:
```
your_data_root/solo_audio_resample/erhu/-5MipMQ25cU_2.wav,your_data_root/solo_extract/erhu/-5MipMQ25cU_2,your_data_root/solo_detect/erhu/-5MipMQ25cU_2.npy
```
For each row, it stores the information: AUDIO_PATH,FRAMES_PATH,DETECTION_PATH

### Unsupervised training and testing
Use the following command to train your IAL-CMS model solely using multi-source videos. For unsupervised training, we use the Synthetic-Duet dataset which artificially synthesized set by mixing two solos in MUSIC-Solo. During training, we randomly mix two synthesized videos and separate object sounds to reconstruct the synthesized audios.
```
python main.py --batch_size 8 --num_workers 4 --gpus 0 --num_sanity_val_steps 2 --max_epochs -1 --vis_root Visualization/SyntheticDuet/IAL-CMS --n_visualize 800 --train_dataset SyntheticDuet --algorithm IAL-CMS --lmconfig config/lm/IAL-CMS/AdaversarialTraining/GeometricGANwAVC.yaml --lambda_independence_incresing_step 8000 16000 24000 --lambda_avc_incresing_step 4000 8000 12000 
```
Use the following command to evaluate the 2-mix unsupervised separation performance.
```
python main.py --test "1" --checkpoint path_2_checkpoint  --batch_size 8 --num_workers 4 --gpus 0 --num_sanity_val_steps 2 --max_epochs -1 --vis_root Visualization/Test/SyntheticDuet/IAL-CMS --n_visualize 800 --train_dataset SyntheticDuet --algorithm IAL-CMS --lmconfig config/lm/IAL-CMS/AdaversarialTraining/GeometricGANwAVC.yaml
```

### Supervised training and testing
Use the following command to train your IAL-CMS model using solo videos:
```
python main.py --batch_size 8 --num_workers 4 --gpus 0 --num_sanity_val_steps 2 --max_epochs -1 --vis_root Visualization/MUSIC-Solo/IAL-CMS --n_visualize 800 --train_dataset MUSIC-Solo --algorithm IAL-CMS --lmconfig config/lm/IAL-CMS/OnlySep/OnlySep_Spatial_Channel_AdaIN.yaml  --lambda_avc_incresing_step 4000 8000 12000 
```
Use the following command to evaluate the 2-mix supervised separation performance. It uses the same test set as unsupervised training.
```
python main.py --test "1" --checkpoint path_2_checkpoint --batch_size 8 --num_workers 4 --gpus 0 --num_sanity_val_steps 2 --max_epochs -1 --vis_root Visualization/Test/MUSIC-Solo/IAL-CMS --n_visualize 800 --train_dataset MUSIC-Solo --algorithm IAL-CMS --lmconfig config/lm/IAL-CMS/OnlySep/OnlySep_Spatial_Channel_AdaIN.yaml
```

# Compared with SOTA
Our proposed method achieve SOTA sound separation results in MUSIC and VGGSound instruments subset.
<div align="center">
<table><tr>
<td><img src="Pictures/MUSIC.png" align=center></td>
<td><img src="Pictures/VGGSound.png" align=center ></td>
</tr></table>
</div>

# Sound Separation Visualization
We visualize the separation results compared with [Co-Separation](https://github.com/rhgao/co-separation/tree/master).
<div align="center">
<img src="Pictures/Visualization.png" align=center>
</div>
