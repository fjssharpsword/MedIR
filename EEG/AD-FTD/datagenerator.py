import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random
import math
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pywt
import mne
from sklearn.model_selection import train_test_split

"""
Dataset: AD-FTD, https://openneuro.org/datasets/ds004504/versions/1.0.3
"""

class DatasetGenerator(Dataset):
    def __init__(self, path_to_ann_dir, path_to_eeg_dir):
        """
        Args:
            
        """
        ann_df = pd.read_csv(path_to_ann_dir, sep=',')
        id_list = ann_df['participant_id'].tolist()
        #com_chn = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
        class_mapping = {'C':0, 'A':1, 'F':2}
        lbl_id = ann_df['Group'].map(class_mapping).tolist()
        eeg_list, lbl_list = [], []
        for id, lbl in zip(id_list, lbl_id):
            eeg_path = path_to_eeg_dir + 'derivatives/' + id +'/eeg/' + id + '_task-eyesclosed_eeg.set'
            #eeg_path = path_to_eeg_dir + id +'/eeg/' + id + '_task-eyesclosed_eeg.set'
            #eeg_list.append(eeg_path)
            #chn_path = path_to_eeg_dir + id +'/eeg/' + id + '_task-eyesclosed_channels.tsv'
            #chn_df = pd.read_csv(chn_path, sep='\t')
            #chn_list.append(chn_df['name'].tolist())

            raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
            events_from_annot, _ = mne.events_from_annotations(raw)
            if len(events_from_annot) > 0:
                bags = mne.Epochs(raw, events=events_from_annot, tmin=-0.2, tmax=0.8)
                eeg_list.extend(bags.get_data())
                lbl_list.extend([lbl]*bags.get_data().shape[0])
            
        self.lbl_list = lbl_list
        self.eeg_list = eeg_list

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            bags: 
            {
                key1: [ind1, ind2, ind3],
                key2: [ind1, ind2, ind3, ind4, ind5],
            ... }
            bag_lbls:
                {key1: 0,
                 key2: 1,
                ... }
        """
        eeg = self.eeg_list[index]
        lbl = self.lbl_list[index]
        
        #multiple instances
        #chs = self.chn_list[index]
        #X = raw.get_data()  #time domain
        #bags = []
        #for i in range(math.floor(X.shape[1]/512)-1):
        #    bags.append(X[:, i*512:(i+1)*512])
        #X = bags

        #X = np.fft.fft(X, n=512, axis=1) #Fourier transform, frequence domian
        
        #X_cA, X_cD = pywt.dwt(X, 'haar', mode='symmetric', axis=1) #wavelet transform, time-frequence domain
        #X = np.concatenate((X_cA, X_cD), axis=1)

        X = torch.FloatTensor(eeg)  #.permute(1,0,2)#.unsqueeze(0)
        y = torch.as_tensor(lbl, dtype=torch.long)

        return X, y

    def __len__(self):
        return len(self.eeg_list)
    
def collate_fn_def(batch):
    return tuple(zip(*batch))

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)

class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        #batch = map(lambda(x, y): (pad_tensor(x, pad=max_len, dim=self.dim), y), batch)
        xs, ys = torch.FloatTensor(), []
        for item in batch:
            xs = torch.cat((xs, pad_tensor(item[0], pad=max_len, dim=self.dim).unsqueeze(0)), 0)
            ys.append(item[1])
        
        ys = torch.LongTensor(ys)
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)
    
def get_dataset(batch_size, shuffle, num_workers, dst_type='tr'):
    PATH_TO_DST_ROOT = '/data/pycode/MedIR/EEG/AD-FTD/dsts/'
    eeg_dataset = DatasetGenerator(path_to_ann_dir=PATH_TO_DST_ROOT + dst_type +'.csv', path_to_eeg_dir='/data/fjsdata/EEG/AD-FTD/')
    #eeg_dataloader = DataLoader(dataset=eeg_dataset, collate_fn=collate_fn_def, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    eeg_dataloader = DataLoader(dataset=eeg_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return eeg_dataloader

def split_inter_patients(test_rate=0.2):
    #parameters:
    ann_files = '/data/fjsdata/EEG/AD-FTD/participants.tsv'
    ann_df = pd.read_csv(ann_files, sep='\t') #header=0, index_col='id'
    train, test = train_test_split(ann_df, test_size=test_rate)
    train.to_csv('/data/pycode/MedIR/EEG/AD-FTD/dsts/tr.csv', index=False)
    test.to_csv('/data/pycode/MedIR/EEG/AD-FTD/dsts/te.csv', index=False)

if __name__ == "__main__":
    #split datasets
    #split_inter_patients(test_rate=0.2)
    #for debug   
    eeg_dst = get_dataset(batch_size=8, shuffle=True, num_workers=0, dst_type='tr')
    for idx, (eeg, lbl) in enumerate(eeg_dst):
        print(eeg.shape)
        print(lbl.shape)
        break
    
    #nohup python3 datagenerator.py > /data/tmpexec/tb_log/datagenerator.log 2>&1 &