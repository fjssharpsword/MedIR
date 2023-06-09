import mne
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import FastICA
import pyedflib
import os

def collection_file(dir):
    file_dict = {}
    for root, _, files in os.walk(dir):
        for file in files:
            ex_name = os.path.splitext(file)[1]
            if ex_name in ['.edf', '.tse']:
                file_name = os.path.splitext(file)[0]
                file_path = os.path.join(root, file)
                if file_name in file_dict.keys():
                    cur_file = file_dict[file_name][0]
                    cur_ext = os.path.splitext(cur_file)[1]
                    if cur_ext=='.edf' and ex_name == '.tse': 
                        file_dict[file_name].append(file_path) #edf file lies first
                    elif cur_ext =='.tse' and ex_name == '.edf': 
                        file_dict[file_name].insert(0,file_path) #edf file lies first
                    else: 
                        print('Collecting file error:' + file_name)
                        file_dict.pop(file_name)
                else:
                    file_dict[file_name] = [file_path]

    return file_dict

def parse_lbl(ann_path):
    ann_list = []
    with open(ann_path, 'r') as ann_file:
        for line in ann_file.readlines()[2:]:
            line = line.split(' ')
            ann_list.append([eval(line[0]), eval(line[1]), line[2]])
    return ann_list

def build_patch(session_dict, fq=250, n_win=2):
    sym_sz = ['bckg', 'cpsz', 'spsz', 'tnsz', 'mysz', 'tcsz', 'gnsz', 'fnsz', 'absz']
    montage_list = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'A1-T3', 'T3-C3', 'C3-CZ',\
               'CZ-C4', 'C4-T4', 'T4-A2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']
    win_len = fq * n_win
    seg, lbl = [], []
    for edf_file, lbl_file in session_dict.values():
        #calculating bipolar
        raw = mne.io.read_raw_edf(edf_file, preload=True) #load edf 
        #raw.filter(l_freq=1, h_freq=70) #filter
        dHz = int(raw.info['sfreq']) #frequence of digital sampling
        if dHz > fq: #down sampling
            raw.resample(fq, npad="auto")
        ref_eeg = raw.get_data() #numpy data
        ch_names = raw.info['ch_names'] #electrodes
        bi_eeg = []
        for montage in montage_list:
            bi_ch = montage.split("-", 1)
            st_i, ed_i = -1, -1
            for i, ele in enumerate(ch_names):
                if bi_ch[0] in ele: st_i = i
                if bi_ch[1] in ele: ed_i = i
            if st_i !=-1 and ed_i !=-1:
                bi_eeg.append(ref_eeg[st_i,:] - ref_eeg[ed_i,:]) #electrode differences
        bi_eeg = np.array(bi_eeg)
        assert bi_eeg.shape[0] == len(montage_list)
        #windows according to labels
        ann_list = parse_lbl(lbl_file) #obtain lables
        if len(ann_list) == 1: continue #only bckg
        for st, ed, sz in ann_list:

            if sym_sz.index(sz) == 0 and lbl.count(0)/(len(lbl)+1)>0.30: #keep non-sz balance
                continue 
            
            st, ed = math.floor(st * fq), math.ceil(ed * fq)
            num = int((ed-st)/win_len)
            if num > 0: 
                for i in range(num):
                    seg.append(bi_eeg[:, st+i*win_len: st+(i+1)*win_len])
                    lbl.append(sym_sz.index(sz))
    #lbl = sym
    #sym = list(set(sym)) #unique key
    #lbl = [sym.index(type) for type in lbl]
    return np.array(seg), np.array(lbl)

def build_dataset():

    #test dataset
    te_dir = '/data/fjsdata/EEG/TUH_EEG/TUSZ/edf/dev/01_tcp_ar/'
    te_dict = collection_file(te_dir)
    print('Total files of test set: {}'.format(len(te_dict)))
    te_seg, te_lbl = build_patch(te_dict)

    print(te_seg.shape)
    print(te_lbl.shape)
    np.save('/data/pycode/MedIR/EEG/TUSZ/dsts/tusz_te_eeg.npy', te_seg)
    np.save('/data/pycode/MedIR/EEG/TUSZ/dsts/tusz_te_lbl.npy', te_lbl)

    #train dataset
    tr_dir = '/data/fjsdata/EEG/TUH_EEG/TUSZ/edf/train/01_tcp_ar/'
    tr_dict = collection_file(tr_dir)
    print('Total files of train set: {}'.format(len(tr_dict)))
    tr_seg, tr_lbl = build_patch(tr_dict)

    print(tr_seg.shape)
    print(tr_lbl.shape)
    np.save('/data/pycode/MedIR/EEG/TUSZ/dsts/tusz_tr_eeg.npy', tr_seg)
    np.save('/data/pycode/MedIR/EEG/TUSZ/dsts/tusz_tr_lbl.npy', tr_lbl)

    print('Class distribution of testset')
    print(pd.Series(te_lbl).value_counts())
    print('Class distribution of trainset')
    print(pd.Series(tr_lbl).value_counts())

def main():
    build_dataset()

if __name__ == "__main__":
    main()
    #nohup python3 -u generator.py >> /data/tmpexec/tb_log/tusz_generator.log 2>&1 &