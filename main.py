import os
from dataset_maker import create_augmented_images
import albumentations as A

NEWDIR = "../augmented_yolov5_dataset/"
OLDDIR = "../yolov5_dataset"
MASKDIR = '../yolov5_dataset/SegmentationObject'
DATA_INFO = '../yolov5_dataset/manual_annotated_data.csv'

Resize = A.Resize(height = 500, width = 500, interpolation=1, p=1.0)
CenterCrop = A.Compose([A.CenterCrop(height = 200, width = 200, p=1.0),Resize])
    
create_augmented_images(augmentation = CenterCrop, olddir = OLDDIR, newdir = NEWDIR, maskdir = MASKDIR, data_info_path = DATA_INFO)
os.system('python3 train.py --data dataset.yaml --hyp custom_hyp.yaml --weights yolov5s.py --cache --epochs 300')
