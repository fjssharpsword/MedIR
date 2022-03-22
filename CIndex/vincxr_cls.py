import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
import time
import random
import re
import sys
import scipy
import SimpleITK as sitk
import pydicom
from scipy import ndimage as ndi
import PIL.ImageOps 
from sklearn.utils import shuffle
import shutil
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import cv2
from pycocotools import mask as coco_mask
import pickle
"""
Dataset: VinBigData Chest X-ray Abnormalities Detection
https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data
1) 150,000 X-ray images with disease labels and bounding box
2) Label:['Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
        'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis', 'No Finding']
"""
#generate 
#https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
#https://github.com/pytorch/vision
class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_dataset_file, bin_keys):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        annotations = pd.read_csv(path_to_dataset_file, sep=',')
        annotations.fillna(0, inplace = True)
        annotations.loc[annotations["class_id"] == 14, ['x_max', 'y_max']] = 1.0
        annotations["class_id"] = annotations["class_id"] + 1
        annotations.loc[annotations["class_id"] == 15, ["class_id"]] = 0
        """
        #first split trainset and testset
        ann_normal = annotations[annotations.class_name=='No finding'].reset_index(drop=True)#"No finding"
        ann_normal = list(set(ann_normal['image_id'].values.tolist()))
        train_size = int(0.8 * len(ann_normal))#8:2
        train_keys = random.sample(ann_normal, train_size)
        test_keys = list(set(ann_normal).difference(set(train_keys)))
        with open("/data/pycode/SFConv/dsts/trKeys_normal.txt", "wb") as fp:   #Pickling
            pickle.dump(train_keys, fp)
        with open("/data/pycode/SFConv/dsts/teKeys_normal.txt", "wb") as fp:   #Pickling
            pickle.dump(test_keys, fp)
        """
        self.CLASS_NAMES = ['No finding', 'Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
                            'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
        annotations = annotations.values #dataframe -> numpy
        image_list = []
        label_list = []
        for annotation in annotations:
            key = annotation[0].split(os.sep)[-1] 
            lbl = int(annotation[2]) #label
            if key in bin_keys:
                image_list.append(key)
                label_list.append(int(lbl))
                    
        self.image_dir = path_to_img_dir
        self.image_list = image_list
        self.label_list = label_list
        
    def _transform_tensor(self, img):
        transform_seq = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
        return transform_seq(img)

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        #show samples
        #print(pd.value_counts(self.label_list))
        #for i in range(len(self.CLASS_NAMES)):
        #    print('The number of {} is {}'.format(self.CLASS_NAMES[i], np.array(self.label_list)[:,i].sum()))
        #image
        key = self.image_list[index]
        img_path = self.image_dir + key + '.jpeg'
        image = Image.open(img_path).convert("RGB")
        image = self._transform_tensor(image)
        #label
        label = self.label_list[index]
        label = torch.as_tensor(label, dtype=torch.long)
        
        return image, label

    def __len__(self):
        return len(self.image_list)

def get_box_dataloader_VIN(batch_size, shuffle, num_workers):
    vin_csv_file = '/data/pycode/SFSAttention/dsts/train.csv'
    vin_image_dir = '/data/fjsdata/Vin-CXR/train_val_jpg/'
  
    if shuffle==True: 
        with open("/data/pycode/SFSAttention/dsts/trKeys.txt", "rb") as fp:   # Unpickling
            key_subset = pickle.load(fp)
        with open("/data/pycode/SFSAttention/dsts/trKeys_normal.txt", "rb") as fp:   # Unpickling
            key_subset_normal = pickle.load(fp)
    else:
        with open("/data/pycode/SFSAttention/dsts/teKeys.txt", "rb") as fp:   # Unpickling
            key_subset = pickle.load(fp)
        with open("/data/pycode/SFSAttention/dsts/teKeys_normal.txt", "rb") as fp:   # Unpickling
            key_subset_normal = pickle.load(fp)

    dataset_box = DatasetGenerator(path_to_img_dir=vin_image_dir, path_to_dataset_file=vin_csv_file, bin_keys=key_subset+key_subset_normal)
    data_loader_box = DataLoader(dataset=dataset_box, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return data_loader_box

if __name__ == "__main__":

    #for debug   
    data_loader_box = get_box_dataloader_VIN(batch_size=8, shuffle=True, num_workers=0)
    for batch_idx, (image, label) in enumerate(data_loader_box):
        print(len(image))
        print(len(label))
        break
