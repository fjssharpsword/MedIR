import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from skimage.measure import label as skilabel
from skimage.measure import regionprops
import os
import pandas as pd
import numpy as np
import time
import random
import sys
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import PIL.ImageOps
from sklearn.model_selection import train_test_split

"""
Dataset: Indian Diabetic Retinopathy Image Dataset (IDRiD)
https://idrid.grand-challenge.org/
Link to access dataset: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
Data Descriptor: http://www.mdpi.com/2306-5729/3/3/25
First Results and Analysis: https://doi.org/10.1016/j.media.2019.101561
A. Localization: center pixel-locations of optic disc and fovea center for all 516 images;
B. Disease Grading: 516 images, 413(80%)images for training, 103(20%) images for test.
1) DR (diabetic retinopathy) grading: 0-no apparent retinopathy, 1-mild NPDR, 2-moderate NPDR, 3-Severe NPDR, 4-PDR
2) Risk of DME (diabetic macular edema): 0-no apparent EX(s), 1-Presence of EX(s) outside the radius of one disc diameter form the macula center,
                                        2-Presence of EX(s) within the radius of one disc diameter form the macula center.
C. Segmentation: 
1) 81 DR images, 54 for training and 27 for test.
2) types: optic disc(OD), microaneurysms(MA), soft exudates(SE), hard exudates(EX), hemorrhages(HE).
"""

class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_msk_dir, lesion='MA'):
        """
        Args:
            path_to_img_dir: path to image directory.
            path_to_msk_dir: path to mask directory.
            transform: optional transform to be applied on a sample.
        """
        imageIDs, maskIDs, lableIDs, OD_IDs = [], [], [], []
        for root, dirs, files in os.walk(path_to_img_dir):
            for file in files:
                ID_img = os.path.join(path_to_img_dir + file)
                if os.path.exists(ID_img):
                    file = os.path.splitext(file)[0]
                    ID_OD = os.path.join(path_to_msk_dir + 'OpticDisc/' + file+'_OD.tif')
                    if lesion=='MA':
                        ID_mask = os.path.join(path_to_msk_dir + 'Microaneurysms/' + file+'_MA.tif')
                        if os.path.exists(ID_mask):
                            imageIDs.append(ID_img)
                            maskIDs.append(ID_mask)
                            OD_IDs.append(ID_OD)
                    elif lesion=='HE':
                        ID_mask = os.path.join(path_to_msk_dir + 'Haemorrhages/' + file+'_HE.tif')
                        if os.path.exists(ID_mask):
                            imageIDs.append(ID_img)
                            maskIDs.append(ID_mask)
                            OD_IDs.append(ID_OD)
                    else: pass
        labelIDs = self.label_lesion(maskIDs, OD_IDs) # label lesion
        self.lesion = lesion
        self.imageIDs = imageIDs
        self.maskIDs = maskIDs
        self.lableIDs = labelIDs
        self.transform_seq_image = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
            ])
        self.transform_seq_mask = transforms.Compose([
            transforms.Resize((224,224)),
            ])
        

    def label_lesion(self, maskIDs, OD_IDs):
        lableIDs = [] #diameter of min lesion, numbers of lesion, distribution of lesions
        for index in range(len(maskIDs)):
            mask_lesion = maskIDs[index]
            mask_od = OD_IDs[index]
            #get centroids of OD
            cv_od = cv2.imread(mask_od, cv2.IMREAD_GRAYSCALE)
            lbl_od = skilabel(cv_od, 2) #connectivity=Eight connected
            props = regionprops(lbl_od)
            cen_od = props[1].centroid #tuple (row, col)
            #handle Lesion
            cv_lesion = cv2.imread(mask_lesion, cv2.IMREAD_GRAYSCALE) #binary image, cv2.COLOR_BGR2GRAY
            lbl_cv_lesion = skilabel(cv_lesion, 2)
            props = regionprops(lbl_cv_lesion)
            min_diameter = float('inf')
            num_quadrant = {0:0, 1:0, 2:0, 3:0} #count up four quadrant
            for i in range(1,len(props)):
                diameter = props[i].equivalent_diameter
                if min_diameter>diameter:
                    min_diameter = diameter
                cen_lesion = props[i].centroid#tuple (row, col)
                if cen_lesion[0] < cen_od[0] and cen_lesion[1] < cen_od[1]:
                    num_quadrant[0] += 1
                elif cen_lesion[0] > cen_od[0] and cen_lesion[1] < cen_od[1]:
                    num_quadrant[1] += 1
                elif cen_lesion[0] < cen_od[0] and cen_lesion[1] > cen_od[1]:
                    num_quadrant[2] += 1
                elif cen_lesion[0] > cen_od[0] and cen_lesion[1] > cen_od[1]:
                    num_quadrant[3] += 1
                else:
                    pass
            num_dis = 0 
            for i in range(len(num_quadrant)):
                if num_quadrant[i]>0: num_dis+=1
            lableIDs.append([min_diameter, len(props)-1, num_dis])
        return lableIDs

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image = self.imageIDs[index]
        mask= self.maskIDs[index]
        label = self.lableIDs[index] 

        image = self.transform_seq_image(Image.open(image).convert('RGB'))
        mask = torch.FloatTensor(np.array(self.transform_seq_mask(Image.open(mask))))
        label = torch.as_tensor(label, dtype=torch.float32)
  
        return image, mask, label, 

    def __len__(self):
        return len(self.imageIDs)

PATH_TO_IMAGES_DIR_TRAIN = '/data/fjsdata/fundus/IDRID/ASegmentation/Images/TrainingSet/'
PATH_TO_IMAGES_DIR_TEST = '/data/fjsdata/fundus/IDRID/ASegmentation/Images/TestingSet/'
PATH_TO_MASKS_DIR_TRAIN = '/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/'
PATH_TO_MASKS_DIR_TEST = '/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TestingSet/'

def get_train_dataloader(batch_size, shuffle, num_workers, lesion='MA'):
    dataset_train = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_TRAIN, path_to_msk_dir=PATH_TO_MASKS_DIR_TRAIN, lesion=lesion)
    #sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train) #for multi cpu and multi gpu
    #data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, sampler = sampler_train, shuffle=shuffle, num_workers=num_workers, pin_memory=True)                       
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_train

def get_test_dataloader(batch_size, shuffle, num_workers, lesion='MA'):
    dataset_test = DatasetGenerator(path_to_img_dir=PATH_TO_IMAGES_DIR_TEST, path_to_msk_dir=PATH_TO_MASKS_DIR_TEST, lesion=lesion)
    data_loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_test

if __name__ == "__main__":
    #for debug   
    dataloader_train = get_train_dataloader(batch_size=8, shuffle=True, num_workers=0, lesion='HE')
    for batch_idx, (images, masks, labels) in enumerate(dataloader_train):
        print(images.shape)
        print(masks.shape)
        print(labels.shape)
        break
