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
            if ex_name in ['.edf', '.lbl']:
                file_name = os.path.splitext(file)[0]
                file_path = os.path.join(root, file)
                if file_name in file_dict.keys():
                    cur_file = file_dict[file_name][0]
                    cur_ext = os.path.splitext(cur_file)[1]
                    if cur_ext=='.edf' and ex_name == '.lbl': 
                        file_dict[file_name].append(file_path) #edf file lies first
                    elif cur_ext =='.lbl' and ex_name == '.edf': 
                        file_dict[file_name].insert(0,file_path) #edf file lies first
                    else: 
                        print('Collecting file error:' + file_name)
                        file_dict.pop(file_name)
                else:
                    file_dict[file_name] = [file_path]

    return file_dict

def parse_lbl(ann_path, sym_sz):
    mon_dict = {}
    sym_dict = {}
    ann_dict = {}
    with open(ann_path, 'r') as ann_file:
        for line in ann_file.readlines():
            #montage information
            if line.startswith('montage'):
                line = line.split(':')[0].split('=')[1]
                line = line.replace('\n', '').replace('\r', '').replace(' ', '') #0,FP1-F7
                num = int(line.split(',')[0])
                bi_ch = line.split(',')[1]
                mon_dict[num] = bi_ch
            #label
            if line.startswith('symbols'):
                line = line.split('=')[1].replace('\n', '').replace('\r', '').replace(' ', '')
                sym_dict = eval(line)

            if bool(sym_dict) and bool(mon_dict) and line.startswith('label'):
                line = line.split('=')[1]
                #start time, end time, channel
                slot = line.split(',')
                st, ed, ch_no = float(slot[2].replace(' ', '')), float(slot[3].replace(' ', '')), int(slot[4].replace(' ', ''))
                ch_name = mon_dict[ch_no]
                #0-1 grountruth
                one_hot = eval(line[line.find('['): line.find(']')+1])
                idx_list = np.nonzero(one_hot)[0]
                for idx in idx_list:
                    cls_name = sym_dict[idx]
                    if cls_name in sym_sz:
                        if ch_name in ann_dict.keys():
                            ann_dict[ch_name].append((cls_name, st, ed))
                        else:
                            ann_dict[ch_name] = [(cls_name, st, ed)] 
    return ann_dict

def build_patch(session_dict, fq=250, n_win=1):
    sym_sz = ['bckg', 'seiz', 'fnsz', 'gnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'cnsz', 'tcsz', 'atsz', 'mysz', 'nesz']
    win_len = fq * n_win
    seg, lbl = [], []
    for edf_file, lbl_file in session_dict.values():
        raw = mne.io.read_raw_edf(edf_file, preload=True) #load edf 
        #raw.filter(l_freq=1, h_freq=70) #filter
        dHz = int(raw.info['sfreq']) #frequence of digital sampling
        if dHz > fq: #down sampling
            raw.resample(fq, npad="auto")
        np_eeg = raw.get_data() #numpy data
        ch_names = raw.info['ch_names'] #electrodes

        ann_dict = parse_lbl(lbl_file, sym_sz) #obtain lables 
        for key in ann_dict.keys():
            if len(ann_dict[key]) == 1: continue #only bckg

            bi_ch = key.split("-", 1) #two electrodes
            st_i, ed_i = -1, -1
            for i, ele in enumerate(ch_names):
                if bi_ch[0] in ele: st_i = i
                if bi_ch[1] in ele: ed_i = i
            if st_i !=-1 and ed_i !=-1:
                ch_eeg = np_eeg[st_i,:] - np_eeg[ed_i,:] #electrode differences
                ch_eeg = (ch_eeg - np.min(ch_eeg))/(np.max(ch_eeg)-np.min(ch_eeg)) #0-1 normalization
                for cls, st, ed in ann_dict[key]:
                    st, ed = math.floor(st * fq), math.ceil(ed * fq)
                    num = int((ed-st)/win_len)
                    if num > 0: 
                        for i in range(num):
                            seg.append(ch_eeg[st+i*win_len: st+(i+1)*win_len])
                            lbl.append(sym_sz.index(cls))

    return np.array(seg), np.array(lbl)

def build_dataset():

    #test dataset
    te_dir = '/data/fjsdata/EEG/TUH_EEG/TUSZ/edf/dev/'
    te_dict = collection_file(te_dir)
    print('Total files of test set: {}'.format(len(te_dict)))
    te_seg, te_lbl = build_patch(te_dict)

    print(te_seg.shape)
    print(te_lbl.shape)
    print(te_lbl.sum())
    np.save('/data/pycode/MedIR/EEG/TUSZ/dsts/tusz_te_eeg_1s.npy', te_seg)
    np.save('/data/pycode/MedIR/EEG/TUSZ/dsts/tusz_te_lbl_1s.npy', te_lbl)

    #train dataset
    tr_dir = '/data/fjsdata/EEG/TUH_EEG/TUSZ/edf/train/'
    tr_dict = collection_file(tr_dir)
    print('Total files of train set: {}'.format(len(tr_dict)))
    tr_seg, tr_lbl = build_patch(tr_dict)

    print(tr_seg.shape)
    print(tr_lbl.shape)
    print(tr_lbl.sum())
    np.save('/data/pycode/MedIR/EEG/TUSZ/dsts/tusz_tr_eeg_1s.npy', tr_seg)
    np.save('/data/pycode/MedIR/EEG/TUSZ/dsts/tusz_tr_lbl_1s.npy', tr_lbl)

def main():
    build_dataset()

if __name__ == "__main__":
    main()
    #nohup python3 -u generator.py >> /data/tmpexec/tb_log/tusz_generator.log 2>&1 &