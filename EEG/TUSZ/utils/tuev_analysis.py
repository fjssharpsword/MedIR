import mne
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import FastICA
import pyedflib
import os
import shutil
import sys

def collection_file(dir):
    file_dict = {}
    for root, _, files in os.walk(dir):
        for file in files:
            ex_name = os.path.splitext(file)[1]
            if ex_name in ['.edf', '.rec']:
                file_name = os.path.splitext(file)[0]
                file_path = os.path.join(root, file)
                if file_name in file_dict.keys():
                    cur_file = file_dict[file_name][0]
                    cur_ext = os.path.splitext(cur_file)[1]
                    if cur_ext=='.edf' and ex_name == '.rec': 
                        file_dict[file_name].append(file_path) #edf file lies first
                    elif cur_ext =='.rec' and ex_name == '.edf': 
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
        for line in ann_file.readlines():
            line = line.split(',')
            ch, st, ed, cl =  eval(line[0]), eval(line[1]), eval(line[2]), eval(line[3])
            if cl == 1: #spsw
                ann_list.append([ch, st, ed, cl])
    return ann_list

def copy_edf_rec(session_dict, fq=250, n_win=2):
    #sym_sz = ['spsw', 'gped', 'pled', 'eyem', 'artf', 'bckg'] #1-6
    #montage = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'A1-T3', 'T3-C3', 'C3-CZ',\
     #          'CZ-C4', 'C4-T4', 'T4-A2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']
    #win_len = fq * n_win
    #seg, lbl = [], []

    n_sub = 0
    tgt_dir = '/data/tmpexec/tb_log/tuev_spsw/'
    for edf_file, lbl_file in session_dict.values():
        
        ann_list = parse_lbl(lbl_file) #obtain lables
        if len(ann_list) > 0: 
            try:
                shutil.copy(edf_file, tgt_dir)
                shutil.copy(lbl_file, tgt_dir)
            except IOError as e:
                print("Unable to copy file. %s" % e)
            except:
                print("Unexpected error:", sys.exc_info())
            print("\n File {} copy done! \n".format(edf_file))
            print("\n File {} copy done! \n".format(lbl_file))
            n_sub = n_sub + 1
    print("\n Subject number:{} \n".format(n_sub))
    """
        #calculating bipolar
        raw = mne.io.read_raw_edf(edf_file, preload=True) #load edf 
        dHz = int(raw.info['sfreq']) #frequence of digital sampling
        if dHz > fq: #down sampling
            raw.resample(fq, npad="auto")
        ref_eeg = raw.get_data() #numpy data
        ch_names = raw.info['ch_names'] #electrodes
        spsw_seg = []
        for ch, st, ed, _ in ann_list:
            bi_ch = montage[ch].split("-", 1)
            st_i, ed_i = -1, -1
            for i, ele in enumerate(ch_names):
                if bi_ch[0] in ele: st_i = i
                if bi_ch[1] in ele: ed_i = i
            if st_i !=-1 and ed_i !=-1:
                bi_eeg = ref_eeg[st_i,:] - ref_eeg[ed_i,:] #electrode differences
                st, ed = math.floor(st * fq), math.ceil(ed * fq)
                spsw_seg.append(bi_eeg[st:ed])
    return np.array(seg)
    """

def build_dataset():

    #eval dataset
    te_dir = '/data/fjsdata/EEG/TUH_EEG/TUEV/edf/eval/'
    te_dict = collection_file(te_dir)
    print('Total files of test set: {}'.format(len(te_dict)))
    copy_edf_rec(te_dict)
    #train dataset
    tr_dir = '/data/fjsdata/EEG/TUH_EEG/TUEV/edf/train/'
    tr_dict = collection_file(tr_dir)
    print('Total files of train set: {}'.format(len(tr_dict)))
    copy_edf_rec(tr_dict)

def main():
    build_dataset()

if __name__ == "__main__":
    main()
    #nohup python3 -u tuev_analysis.py >> /data/tmpexec/tb_log/tuev_analysis.log 2>&1 &