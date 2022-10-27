import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from collections import Counter
ImageFile.LOAD_TRUNCATED_IMAGES = True

"""
Dataset: OIA-DDR, https://github.com/nkicsl/DDR-dataset
"""

class DatasetGenerator(Dataset):
    def __init__(self, path_to_img_dir, path_to_lbl_dir):
        """
        Args:
            path_to_img_dir: path to image directory.
            path_to_lbl_dir: path to label directory.
            transform: optional transform to be applied on a sample.
        """
        self.CLASS_NAMES = ['No DR', "Mild NPDR", 'Moderate NPDR', 'Severe NPDR', 'PDR']
        imgs_list = []
        lbls_list = []
        with open(path_to_lbl_dir, "r") as f:
            for line in f:
                items = line.split(' ')
                img_name =  os.path.join(path_to_img_dir, items[0])
                lbl_idx = int(items[1])
                if os.path.exists(img_name) and lbl_idx != 5: #remove class ungradable
                    imgs_list.append(img_name)
                    lbls_list.append(int(items[1]))
                    #lbl_onehot = np.zeros(len(self.CLASS_NAMES)) #one-hot
                    #lbl_onehot[int(items[1])] = 1
                    #lbls_list.append(lbl_onehot)
                    
        self.imgs_list = imgs_list
        self.lbls_list = lbls_list
        #print(Counter(lbls_list))
        self.transform_seq = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        img_name = self.imgs_list[index]
        img_cwh = Image.open(img_name).convert('RGB')
        lbl_idx = self.lbls_list[index]
        if self.transform_seq is not None:
            img_cwh = self.transform_seq(img_cwh)
        #lbl_idx = torch.as_tensor(lbl_idx, dtype=torch.float32) #for bce loss, with one-hot
        lbl_idx = torch.as_tensor(lbl_idx, dtype=torch.long) #for ce loss
        return img_cwh, lbl_idx

    def __len__(self):
        return len(self.imgs_list)

PATH_TO_DST_ROOT = '/data/fjsdata/fundus/OIA-DDR/DR_Grading/'

def get_fundus_DDR(batch_size, shuffle, num_workers, dst_type='train'):
    dataset_ddr = DatasetGenerator(path_to_img_dir=PATH_TO_DST_ROOT + dst_type +'/', path_to_lbl_dir=PATH_TO_DST_ROOT+dst_type+'.txt')
    data_loader_ddr = DataLoader(dataset=dataset_ddr, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader_ddr

if __name__ == "__main__":
    #for debug   
    ddr_dst = get_fundus_DDR(batch_size=10, shuffle=True, num_workers=0, dst_type='test')
    for batch_idx, (img, lbl) in enumerate(ddr_dst):
        print(img.shape)
        print(lbl)
        break