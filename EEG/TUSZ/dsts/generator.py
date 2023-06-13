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
            if ex_name in ['.edf', '.csv_bi']:
                file_name = os.path.splitext(file)[0]
                file_path = os.path.join(root, file)
                if file_name in file_dict.keys():
                    cur_file = file_dict[file_name][0]
                    cur_ext = os.path.splitext(cur_file)[1]
                    if cur_ext=='.edf' and ex_name == '.csv_bi': 
                        file_dict[file_name].append(file_path) #edf file lies first
                    elif cur_ext =='.csv_bi' and ex_name == '.edf': 
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
        for line in ann_file.readlines()[6:]:
            line = line.split(',')
            _, st, ed, cl =  line[0], eval(line[1]), eval(line[2]), line[3]
            if cl == 'seiz':
                ann_list.append([st, ed])
    return ann_list

def build_patch(session_dict, fq=250, n_win=2):
    #sym_sz = ['bckg', 'seiz']
    montage = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'A1-T3', 'T3-C3', 'C3-CZ',\
               'CZ-C4', 'C4-T4', 'T4-A2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']

    win_len = fq * n_win
    eegs, lbls = [], []
    for edf_file, lbl_file in session_dict.values():
        ann_list = parse_lbl(lbl_file) #obtain lables

        if len(ann_list) > 0: 
            raw = mne.io.read_raw_edf(edf_file, preload=True) #load edf 
            #raw.filter(l_freq=1, h_freq=70) #filter
            dHz = int(raw.info['sfreq']) #frequence of digital sampling
            if dHz > fq: #down sampling
                raw.resample(fq, npad="auto")
            ref_eeg = raw.get_data() #numpy data
            ch_names = raw.info['ch_names'] #electrodes

            bi_eeg = []
            for bi_ch in montage:
                bi_ch =bi_ch.split("-", 1)
                st_i, ed_i = -1, -1
                for i, ele in enumerate(ch_names):
                    if bi_ch[0] in ele: st_i = i
                    if bi_ch[1] in ele: ed_i = i
                if st_i !=-1 and ed_i !=-1:
                    bi_eeg.append(ref_eeg[st_i,:] - ref_eeg[ed_i,:]) #electrode differences
            bi_eeg = np.array(bi_eeg)

            bi_lbl = np.zeros(bi_eeg.shape[1])
            if bi_eeg.shape[0] == len(montage):
                for st, ed in ann_list:
                    st, ed = math.floor(st * fq), math.ceil(ed * fq)
                    bi_lbl[st:ed] = 1
            
                tokens = np.where(np.diff(bi_lbl != 0))[0] + 1
                tokens = np.insert(tokens, 0, 0)
                tokens = np.append(tokens, len(bi_lbl))
                for p in range(0, len(tokens)-1, 1):
                    num = int((tokens[p+1]-tokens[p])/win_len)
                    for j in range(num):
                        eegs.append(bi_eeg[:, tokens[p]+j*win_len:tokens[p]+(j+1)*win_len])
                        if bi_lbl[tokens[p]+j*win_len:tokens[p]+(j+1)*win_len].sum()>0: 
                            lbls.append(1)
                        else:
                            lbls.append(0)

    return np.array(eegs), np.array(lbls)

def build_dataset():
    """
    #test dataset
    te_dir = '/data/fjsdata/EEG/TUH_EEG/TUSZ/edf/dev/'
    te_dict = collection_file(te_dir)
    print('Total files of test set: {}'.format(len(te_dict)))
    te_seg, te_lbl = build_patch(te_dict)

    print(te_seg.shape)
    print(te_lbl.shape)
    print('Class distribution of testset')
    print(pd.Series(te_lbl).value_counts())
    np.save('/data/pycode/MedIR/EEG/TUSZ/dsts/tusz_te_eeg.npy', te_seg)
    np.save('/data/pycode/MedIR/EEG/TUSZ/dsts/tusz_te_lbl.npy', te_lbl)

    #eval dataset
    ev_dir = '/data/fjsdata/EEG/TUH_EEG/TUSZ/edf/eval/'
    ev_dict = collection_file(ev_dir)
    print('Total files of evaluation set: {}'.format(len(ev_dict)))
    ev_seg, ev_lbl = build_patch(ev_dict)

    print(ev_seg.shape)
    print(ev_lbl.shape)
    print('Class distribution of evalset')
    print(pd.Series(ev_lbl).value_counts())
    np.save('/data/pycode/MedIR/EEG/TUSZ/dsts/tusz_ev_eeg.npy', ev_seg)
    np.save('/data/pycode/MedIR/EEG/TUSZ/dsts/tusz_ev_lbl.npy', ev_lbl)
    """
    #train dataset
    tr_dir = '/data/fjsdata/EEG/TUH_EEG/TUSZ/edf/train/'
    tr_dict = collection_file(tr_dir)
    print('Total files of train set: {}'.format(len(tr_dict)))
    tr_seg, tr_lbl = build_patch(tr_dict)

    print(tr_seg.shape)
    print(tr_lbl.shape)
    print('Class distribution of trainset')
    print(pd.Series(tr_lbl).value_counts())
    np.save('/data/pycode/MedIR/EEG/TUSZ/dsts/tusz_tr_eeg.npy', tr_seg)
    np.save('/data/pycode/MedIR/EEG/TUSZ/dsts/tusz_tr_lbl.npy', tr_lbl)
    
def main():
    build_dataset()

if __name__ == "__main__":
    main()
    #nohup python3 -u generator.py >> /data/tmpexec/tb_log/tusz_generator.log 2>&1 &