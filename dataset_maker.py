import albumentations as A
import pathlib
from skimage.measure import regionprops, regionprops_table,find_contours
from skimage.measure import label as skinmage_label
from skimage.io import imread
import shutil
import pandas as pd
from sklearn import preprocessing
import numpy as np

def create_augmented_images(augmentation, olddir, newdir, maskdir, data_info_path):
    #create an empty director of new path with image/ train and image/test
    newtrain = pathlib.Path(newdir,'images', 'train')
    newtest = pathlib.Path(newdir, 'images', 'train')
    odltrain = pathlib.Path(olddir,'images', 'train')
    oldtest = pathlib.Path(olddir, 'images', 'test')
    #read all images as files names into a varaible
    org_testimages = list(oldtest.glob('*.jpg')) 
    org_trainimages = list(odltrain.glob('*.jpg')) 
    

    #Preprocessing labels
    data_info = pd.read_csv(data_info_path, index_col=False)
    le = preprocessing.LabelEncoder()
    label = data_info['name']
    le.fit(label)
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    data_info['label'] = le.transform(label)
    #AUGMENT FOR TRAIN SET
    for img_paths in [org_trainimages, org_testimages]:
        for img_path in img_paths:
            img = imread(img_path)        
            #MASK
            image_name = img_path.stem
            mask_path = get_mask_path(image_name, maskdir)
            mask = imread(mask_path)
            #AUGMENTATION
            augmented = augmentation(image = img, mask = mask)
            augmented_mask = augmented['mask']
            augmented_img = augmented['image']
            #Process mask
            coords  = get_bbox(augmented_mask)
            cat_label = get_label(data_info, img_path.name)
            label = le.transform([cat_label])[0]
            bbox = [np.insert(coord,0, label) for coord in coords]
            #NEW DIR
            img_newpath = pathlib.Path(newdir, 'images', image_name + '.jpg')
            mask_newpath = pathlib.Path(maskdir, 'labels', image_name + '.txt')
            #save file
            np.savetxt(mask_newpath, bbox, fmt='%d %.5f %.5f %.5f %.5f')
            cv2.imwrite(img_newpath, augmented_img)
        
        
def get_label(data_info, name):
    return data_info[data_info.file == name].name.iloc[0]
    
def get_mask_path(image_name, maskdir):
    pic_png = image_name + '.png'
    mask_path = pathlib.Path(maskdir, pic_png)
    return mask_path

def get_bbox(mask): 
    output = []
    for prop in regionprops(skinmage_label(mask)):
            width = mask.shape[1]
            height = mask.shape[0]
            x1, y1 = prop.bbox[1], prop.bbox[0]
            x2, y2 = prop.bbox[4],prop.bbox[3]
            x = (x1 + x2)//2
            x = x / width
            y = (x1 + x2)//2
            y = y / height
            h = y2 - y1
            h = h / height
            w = x2 - x1
            w = w / width
            seg = [x,y,h,w]
            output.append(seg)
    return output
