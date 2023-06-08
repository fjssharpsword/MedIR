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

def parse_lbl(ann_path):
    sym_sz = ['spsw', 'seiz', 'fnsz', 'gnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'cnsz', 'tcsz', 'atsz', 'mysz', 'nesz']
    mon_dict = {}
    sym_dict = {}
    flag = False
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
                #0-1 grountruth
                one_hot = eval(line[line.find('['): line.find(']')+1])
                idx_list = np.nonzero(one_hot)[0]
                for idx in idx_list:
                    cls_name = sym_dict[idx]
                    if cls_name in sym_sz: #containing seizure waves
                        flag = True
    return flag

def mv_file(session_dict, tgt_dir):
    n_sub = 0
    for edf_file, lbl_file in session_dict.values():
        flag = parse_lbl(lbl_file) #obtain lables 
        if flag:    
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

def collect_edf():

    #test dataset
    te_dir = '/data/fjsdata/EEG/TUH_EEG/TUSZ/edf/dev/'
    te_dict = collection_file(te_dir)
    print('Total files of test set: {}'.format(len(te_dict)))
    tgt_dir = '/data/fjsdata/EEG/JNU-SPSW/files2/'
    mv_file(te_dict, tgt_dir)

    #train dataset
    tr_dir = '/data/fjsdata/EEG/TUH_EEG/TUSZ/edf/train/'
    tr_dict = collection_file(tr_dir)
    print('Total files of train set: {}'.format(len(tr_dict)))
    tgt_dir = '/data/fjsdata/EEG/JNU-SPSW/files3/'
    mv_file(tr_dict, tgt_dir)

def main():
    collect_edf()

if __name__ == "__main__":
    main()
    #nohup python3 -u collector.py >> /data/tmpexec/tb_log/collector.log 2>&1 &