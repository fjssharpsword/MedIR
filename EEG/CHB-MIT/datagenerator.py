import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from inter_patient import Patient
import random
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pywt

"""
Dataset: CHB-MIT, https://physionet.org/content/chbmit/1.0.0/
"""

class DatasetGenerator(Dataset):
    def __init__(self, path_to_eeg_dir, path_to_lbl_dir):
        """
        Args:
            
        """
        self.eeg_list = np.load(path_to_eeg_dir)
        self.lbl_list = np.load(path_to_lbl_dir)   

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        eeg = self.eeg_list[index]
        #eeg = np.fft.fft2(eeg, axes=(1,)) #DFT on column
        eeg, _, _, _ = plt.specgram(np.mean(eeg, axis=1), NFFT=2560, Fs=2, noverlap=2558)
        #eeg, _ = pywt.cwt(np.mean(eeg, axis=1), np.arange(1, 512+1), 'cgau8')

        lbl = self.lbl_list[index]

        #eeg = torch.FloatTensor(eeg).permute(1,0)
        eeg = torch.FloatTensor(eeg).unsqueeze(0)
        lbl = torch.as_tensor(lbl, dtype=torch.long)

        return eeg, lbl

    def __len__(self):
        return len(self.eeg_list)

PATH_TO_DST_ROOT = '/data/pycode/MedIR/EEG/CHB-MIT/dsts/'

def get_intra_dataset(batch_size, shuffle, num_workers, dst_type='train'):
    eeg_dataset = DatasetGenerator(path_to_eeg_dir=PATH_TO_DST_ROOT + dst_type +'/eeg.npy', path_to_lbl_dir=PATH_TO_DST_ROOT+dst_type+'/lbl.npy')
    eeg_dataloader = DataLoader(dataset=eeg_dataset, batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return eeg_dataloader

def split_inter_patients(seconds=2, neg_rate=2):
    #parameters:
    #seconds: windows sliding length
    #neg_rate: times of negative samples to positive samples

    patient_ids = [id for id in range(1, 25)]
    patient_ids.remove(12) # delete patient 12 with different channels.
    random.shuffle(patient_ids)
    te_ids, tr_ids = patient_ids[:6], patient_ids[6:]

    #obtain channel names
    ch_com = np.array(['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', \
             'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8'])
    for p_id in patient_ids:
        p = Patient(p_id)
        p_ch_name = p.get_channel_names()
        ch_com = np.intersect1d(ch_com, p_ch_name)
        p.close_files()#release files

    #test set
    sei_data, sei_idx = [], []#positive samples
    non_sei_data, non_sei_idx = [], []#negative samples
    for te_id in te_ids:
        p = Patient(te_id)
        data = p.get_eeg_data(ch_com)
        win_len = int(p.get_sampling_rate() * seconds)

        for sei_seg in p._seizure_intervals:
            num = int((sei_seg[1]-sei_seg[0])/win_len)
            pos = sei_seg[0]
            for i in range(num):
                pos = sei_seg[0]+i*win_len
                sei_data.append(data[pos:pos+win_len])
                sei_idx.append(1)

        for i in range(len(p._seizure_intervals)-1):
            num = int((p._seizure_intervals[i+1][0]-p._seizure_intervals[i][1])/win_len)
            for j in range(num):
                pos = p._seizure_intervals[i][1]+j*win_len
                non_sei_data.append(data[pos:pos+win_len])
                non_sei_idx.append(0)
                if len(non_sei_data) >= len(sei_data)*neg_rate: 
                    break

        p.close_files()#release files
    X_te = np.array(sei_data + non_sei_data)
    y_te = np.array(sei_idx + non_sei_idx)
    np.save(PATH_TO_DST_ROOT+'test/eeg.npy', X_te)
    np.save(PATH_TO_DST_ROOT+'test/lbl.npy', y_te)
            
    #trian set
    sei_data, sei_idx = [], []#positive samples
    non_sei_data, non_sei_idx = [], []#negative samples
    for tr_id in tr_ids:
        p = Patient(tr_id)
        data = p.get_eeg_data(ch_com)
        win_len = int(p.get_sampling_rate() * seconds)

        for sei_seg in p._seizure_intervals:
            num = int((sei_seg[1]-sei_seg[0])/win_len)
            pos = sei_seg[0]
            for i in range(num):
                pos = sei_seg[0]+i*win_len
                sei_data.append(data[pos:pos+win_len])
                sei_idx.append(1)

        for i in range(len(p._seizure_intervals)-1):
            if len(non_sei_data) >= len(sei_data)*neg_rate: 
                break
            num = int((p._seizure_intervals[i+1][0]-p._seizure_intervals[i][1])/win_len)
            for j in range(num):
                pos = p._seizure_intervals[i][1]+j*win_len
                non_sei_data.append(data[pos:pos+win_len])
                non_sei_idx.append(0)
                if len(non_sei_data) >= len(sei_data)*neg_rate: 
                    break

        p.close_files()#release files

    X_tr = np.array(sei_data + non_sei_data)
    y_tr = np.array(sei_idx + non_sei_idx)
    np.save(PATH_TO_DST_ROOT+'train/eeg.npy', X_tr)
    np.save(PATH_TO_DST_ROOT+'train/lbl.npy', y_tr)

def inter_patients_CV(seconds=2, neg_rate=2):

    patient_ids = [id for id in range(1, 25)]
    patient_ids.remove(12) # delete patient 12 with different channels.
    random.shuffle(patient_ids)

    #obtain channel names
    ch_com = np.array(['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', \
             'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8'])
    for p_id in patient_ids:
        p = Patient(p_id)
        p_ch_name = p.get_channel_names()
        ch_com = np.intersect1d(ch_com, p_ch_name)
        p.close_files()#release files

    #datasets
    sei_data, sei_idx = [], []#positive samples
    non_sei_data, non_sei_idx = [], []#negative samples
    for p_id in patient_ids:
        win_len = int(p.get_sampling_rate() * seconds)
        p = Patient(p_id)
        data = p.get_eeg_data(ch_com)

        for sei_seg in p._seizure_intervals:
            num = int((sei_seg[1]-sei_seg[0])/win_len)
            pos = sei_seg[0]
            for i in range(num):
                pos = sei_seg[0]+i*win_len
                sei_data.append(data[pos:pos+win_len])
                sei_idx.append(1)

        for i in range(len(p._seizure_intervals)-1):
            num = int((p._seizure_intervals[i+1][0]-p._seizure_intervals[i][1])/win_len)
            for j in range(num):
                pos = p._seizure_intervals[i][1]+j*win_len
                non_sei_data.append(data[pos:pos+win_len])
                non_sei_idx.append(0)
                if len(non_sei_data) >= len(sei_data)*neg_rate: 
                    break

        p.close_files()#release files

    X = np.array(sei_data + non_sei_data)
    y = np.array(sei_idx + non_sei_idx)  
    np.save(PATH_TO_DST_ROOT+'eeg_kfold_500ms.npy', X)
    np.save(PATH_TO_DST_ROOT+'lbl_kfold_500ms.npy', y)

if __name__ == "__main__":
    #split datasets
    #split_inter_patients(seconds=20, neg_rate=2)
    #for debug   
    #eeg_dst = get_intra_dataset(batch_size=2, shuffle=True, num_workers=0, dst_type='train')
    #for idx, (eeg, lbl) in enumerate(eeg_dst):
    #    print(eeg.shape)
    #    print(lbl.shape)
    #    break
    #for k-fold cross validation
    #inter_patients_CV(seconds=2, neg_rate=2)
    inter_patients_CV(seconds=0.5, neg_rate=2)
    #nohup python3 datagenerator.py > /data/tmpexec/tb_log/datagenerator.log 2>&1 &