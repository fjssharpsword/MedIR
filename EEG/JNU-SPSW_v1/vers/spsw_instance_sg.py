import mne
import math
import numpy as np
import pandas as pd
import os
import pickle 
import matplotlib.pyplot as plt

class SPSWInstance:
    def __init__(self, id, down_fq=250):

        #parse edf and csv
        edf_path =  "/data/fjsdata/EEG/JNU-SPSW/files1/" + id + '.edf'
        ann_path =  "/data/fjsdata/EEG/JNU-SPSW/files1/" + id + '.csv'
        raw_np = mne.io.read_raw_edf(edf_path, preload=True)
     
        #filter
        raw_np.filter(l_freq=1, h_freq=70) 
        #raw_np.notch_filter(freqs=50)

        #downsampling
        sfreq = int(raw_np.info['sfreq']) 
        #time_duration = raw_np.n_times / self.sfreq #second
        if  sfreq != down_fq:
            raw_np.resample(down_fq, npad="auto") 

        #get SPSW instance
        ann_dict = self._parse_annotation(ann_path)
        self.eeg_dict = self._init_instance(raw_np, ann_dict, down_fq)

    def _init_instance(self, raw_np, ann_dict, down_fq):

        eeg_data = raw_np.get_data() #numpy data
        ch_names = raw_np.info['ch_names'] #electrodes
        eeg_dict = {}
        for key in ann_dict.keys(): 
            bi_ch = key.split("-", 1) #two electrodes
            F_idx, S_idx = -1, -1
            for i, ele in enumerate(ch_names):
                if bi_ch[0] in ele: F_idx = i
                if bi_ch[1] in ele: S_idx = i
            if F_idx !=-1 and S_idx !=-1:
                ch_eeg = eeg_data[F_idx,:] - eeg_data[S_idx,:]
                ch_lbl = np.zeros(len(ch_eeg))
                values = ann_dict[key]
                for val in values:
                    st, ed = math.floor(val[0] * down_fq), math.ceil(val[1] * down_fq)
                    ch_lbl[st:ed] = 1
                eeg_dict[key] = (ch_eeg, ch_lbl)
        return eeg_dict

    def _parse_annotation(self, ann_path):

        ann_dict = {}
        with open(ann_path, 'r') as ann_file:
            for line in ann_file.readlines()[4:]:
                row = line.replace('\n', '').replace('\r', '').replace(' ', '').split(",")
                chs, st, ed = row[1], eval(row[2]), eval(row[3])
                if chs in ann_dict.keys():
                    ann_dict[chs].append((st,ed))
                else:
                    ann_dict[chs] = [(st,ed)]

        return ann_dict
    
def build_dataset():
    dir = "/data/fjsdata/EEG/JNU-SPSW/files1/"
    ids = []
    for _, _, files in os.walk(dir):
        for file in files:
            name = os.path.splitext(file)[0]
            if name not in ids:
                ids.append(name)

    eeg_dict = {}
    for id in ids:
        spsw = SPSWInstance(id)
        for key in spsw.eeg_dict.keys():
            if key in eeg_dict.keys():
                eeg_dict[key].append(spsw.eeg_dict[key])
            else:
                eeg_dict[key] = [spsw.eeg_dict[key]]

    dir = "/data/pycode/MedIR/EEG/JNU-SPSW/dsts/"
    with open(dir+'eeg_sg.pkl', 'wb') as f:
        pickle.dump(eeg_dict, f)

def read_dict_from_file():

    dir = "/data/pycode/MedIR/EEG/JNU-SPSW/dsts/"
    with open(dir+'eeg_sg.pkl', 'rb') as f:
        eeg_dict = pickle.load(f)
        #print(eeg_dict['FP1-F3'][0][0].shape)
        #print(eeg_dict['FP1-F3'][0][1].sum())

    min_eeg, max_lbl = float('inf'), 0
    for key in eeg_dict.keys():
        values = eeg_dict[key]
        for val in values:
            eeg = val[0]
            lbl = val[1]
            assert eeg.shape == lbl.shape
            if min_eeg > eeg.shape[0]: min_eeg = eeg.shape[0]
            if max_lbl < lbl.sum(): max_lbl = lbl.sum()
    print(min_eeg)#5500
    print(max_lbl)#91110

def vis_sequence():
    dir = "/data/pycode/MedIR/EEG/JNU-SPSW/dsts/"
    with open(dir+'eeg_sg.pkl', 'rb') as f:
        eeg_dict = pickle.load(f)

    fig, axes = plt.subplots(1,2, constrained_layout=True,figsize=(12,6))
    win_len = 500 #2sx250Hz
    for i in range(2):   
        key = list(eeg_dict.keys())[i+12]  
        vals = eeg_dict[key]
        eeg, lbl = vals[0][0], vals[0][1]

        num = math.floor(len(lbl)/win_len)
        for j in range(num):
            lbl_seg = lbl[j*win_len:(j+1)*win_len]
            if lbl_seg.sum()>0:
                eeg_seg = eeg[j*win_len:(j+1)*win_len]
                eeg_seg = (eeg_seg - np.min(eeg_seg))/(np.max(eeg_seg)-np.min(eeg_seg))

                x = [id for id in range(1, len(lbl_seg)+1)]
                axes[i].plot(x, eeg_seg, color = 'g', label=key)

                st, ed = np.argwhere(lbl_seg==1)[0], np.argwhere(lbl_seg==1)[-1]
                axes[i].plot(st, eeg_seg[st], marker='^', color='r')
                axes[i].plot(ed, eeg_seg[ed], marker='v', color='r')

                axes[i].legend(loc = "best")
                axes[i].grid(b=True, ls=':')

                break
    fig.savefig('/data/pycode/MedIR/EEG/JNU-SPSW/imgs/eeg_time_seq.png', dpi=300, bbox_inches='tight') 
        

if __name__ == "__main__":

    #build_dataset()
    #read_dict_from_file()
    vis_sequence()

    
            
    
