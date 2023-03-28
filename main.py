import os
from dataset_maker import create_augmented_images
import albumentations as A
import time

NEWDIR = "../augmented_yolov5_dataset/"
OLDDIR = "../yolov5_dataset"
MASKDIR = '../yolov5_dataset/SegmentationObject'
DATA_INFO = '../yolov5_dataset/manual_annotated_data.csv'

Resize = A.Resize(height = 500, width = 500, interpolation=1, p=1.0)
CenterCrop = A.Compose([A.CenterCrop(height = 200, width = 200, p=1.0),Resize])
CenterCrop_2 = A.Compose([A.CenterCrop(height = 100, width = 100, p=1.0),Resize])
Rotation = A.Rotate(p=1.0)
Flip = A.Flip(p=1.0)
##MOTION BLUR
#COLOR AUGMENTATION
CLAHE = A.CLAHE(clip_limit = 16, tile_grid_size=(4, 4), p=1.0)
Sharpen = A.Sharpen(alpha= (0.6,0.8), lightness = (0.6,1.0), p=1.0)
ChannelShuffle = A.ChannelShuffle(p =1)
ColorJitter =  A.ColorJitter(brightness=0.8, hue = 0.42, contrast = 0.54, saturation = 0.65, p=1.0)
ToGray = A.ToGray(p=1)
ToSepia = A.ToSepia(p=1)
GaussNoise = A.GaussNoise(p =1)
Normalize = A.Normalize(mean = (0.184, 1.289, 0.661), std = (0.708, 0.338, 0.177), p = 1.0)
### SOLARIZE

#CUTOUT
GridDropout = A.GridDropout(ratio = 0.6, random_offset = True, p=1)
Cutout = A.CoarseDropout(max_holes=1, p =1, max_height=50, max_width=50)


aug_dict = {}
# # aug_dict['CenterCrop_2_of_5'] = CenterCrop
# # aug_dict['CenterCrop_1_of_5'] = CenterCrop_2
# aug_dict['Rotation'] = Rotation
aug_dict['Normalize'] = Normalize
aug_dict['Flip'] = Flip
# #COLOR
# aug_dict['CLAHE'] = CLAHE
# aug_dict['Sharpen'] = Sharpen
# aug_dict['ChannelShuffle'] = ChannelShuffle
# aug_dict['ToGray'] = ToGray
# aug_dict['ToSepia'] = ToSepia
# aug_dict['GaussNoise'] = GaussNoise

# #Cutout
# aug_dict['GridDropout'] = GridDropout
# aug_dict['Cutout'] = Cutout

MotionBlur = A.MotionBlur(blur_limit = 21, p = 1.0)
# aug_dict['Perspective'] = A.Perspective(scale = 0.3, p = 1.0)
# aug_dict['Solarize'] = A.Solarize(threshold = 192, p = 1)
def Rand_HP(n):
        def RandAugment(image, n = n, m = 10):
        # Define a set of possible image transformations
            transforms_list = [Rotation, 
                            Flip,
                            MotionBlur,
                            Sharpen,
                            CenterCrop,
                            GaussNoise,
                            GridDropout
                            ]
            # Apply a random sequence of n transformations with magnitude m
            aug = A.Compose([transforms_list[i] for i in np.random.choice(len(transforms_list), n)])
            aug_image = aug(image = image.astype(np.uint8))['image']
            output = {}
            output['image'] = aug_image
            #FOR VISUALISATION
            output['augs'] = [str(x).partition('(')[0] for x in aug]
            return output
        return RandAugment
Perspective =  A.Perspective(scale = 0.3, p = 1.0)
Solarize = A.Solarize(threshold = 192, p = 1)

# os.system('python3 train.py --data dataset_baseline.yaml --hyp lr_e3.yaml  --cache --epochs 200 --run_name hp_lr_e3')
# time.sleep(30)
# os.system('python3 train.py --data dataset_baseline.yaml --hyp lr_e4.yaml --weights yolov5s.pt --cache --epochs 100 --run_name hp_lr_e4')
# time.sleep(30)
# os.system('python3 train.py --data dataset_baseline.yaml --hyp lr_e5.yaml --weights yolov5s.pt --cache --epochs 100 --run_name hp_lr_e5')
# time.sleep(30)
# os.system('python3 train.py --data dataset_baseline.yaml --hyp lr_e6.yaml --weights yolov5s.pt --cache --epochs 100 --run_name hp_lr_e6')


create_augmented_images(augmentation = Rand_HP(2), olddir = OLDDIR, newdir = NEWDIR, maskdir = MASKDIR, data_info_path = DATA_INFO, is_normalize = True)
os.system('python3 train.py --data dataset.yaml --hyp custom_hyp.yaml --weights yolov5s.pt --cache --epochs 100 --run_name {}'.format('Randaug'))
time.sleep(60)
create_augmented_images(augmentation = Solarize, olddir = OLDDIR, newdir = NEWDIR, maskdir = MASKDIR, data_info_path = DATA_INFO, is_normalize = True)
os.system('python3 train.py --data dataset.yaml --hyp custom_hyp.yaml --weights yolov5s.pt --cache --epochs 100 --run_name {}'.format('Solarize'))
time.sleep(60)
create_augmented_images(augmentation = MotionBlur, olddir = OLDDIR, newdir = NEWDIR, maskdir = MASKDIR, data_info_path = DATA_INFO, is_normalize = True)
os.system('python3 train.py --data dataset.yaml --hyp custom_hyp.yaml --weights yolov5s.pt --cache --epochs 100 --run_name {}'.format('MotionBlur'))
time.sleep(60)