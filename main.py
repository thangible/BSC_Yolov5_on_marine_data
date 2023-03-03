import os
from dataset_maker import *
import albumentations as A



NEWDIR = "../augmented_yolov5_dataset/"
OLDDIR = "../yolov5_dataset"
MASKDIR = '../yolov5_dataset/SegmentationObject'

Resize = A.Resize(height = 500, width = 500, interpolation=1, p=1.0)
CenterCrop = A.Compose([A.CenterCrop(height = 200, width = 200, p=1.0),Resize])
    
create_augmented_images(augmentation = CenterCrop, old_dir = OLDDIR, new_dir = NEWDIR, mask_dir = MASKDIR)
os.system('python3 train.py --data dataset.yaml --hyp custom_hyp.yaml --weights yolov5s.py --cache --epochs 300')
