import mne
import math
import numpy as np
import pandas as pd
import random
import os
import pickle 
import matplotlib.pyplot as plt
import pywt

def dice_coef(input, target):
    smooth = 1

    N = target.size(0)
    input_flat = input.view(N, -1)
    target_flat = target.view(N, -1)
    
    intersection = input_flat * target_flat
    
    coef = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) +smooth)
    coef = coef.sum() / N
    return coef

class SPSWInstance:
    def __init__(self, id, down_fq=250, seg_len=250*1):

        #parse edf and csv
        edf_path =  "/data/fjsdata/EEG/JNU-SPSW/files2/tuev_spsw/" + id + '.edf'
        ann_path =  "/data/fjsdata/EEG/JNU-SPSW/files2/tuev_spsw/" + id + '.rec'
        raw_np = mne.io.read_raw_edf(edf_path, preload=True)
        #filter
        raw_np.filter(l_freq=1, h_freq=70)

        #downsampling
        sfreq = int(raw_np.info['sfreq']) 
        #time_duration = raw_np.n_times / self.sfreq #second
        if  sfreq != down_fq:
            raw_np.resample(down_fq, npad="auto") 
        self.down_fq = down_fq

        #get SPSW instance
        ann_dict = self._parse_annotation(ann_path)
        self.eeg, self.lbl = self._parse_EEGWave(raw_np, ann_dict, down_fq, seg_len)

    def _parse_EEGWave(self, raw_np, ann_dict, down_fq, seg_len):

        eeg_data = raw_np.get_data() #numpy data
        ch_names = raw_np.info['ch_names'] #electrodes
        eeg, lbl = [], []
        for key in ann_dict.keys(): 

            #screening out specified channels
            #if key not in ['T6-O2']: continue #['FP2-F8']

            bi_ch = key.split("-", 1) #two electrodes
            F_idx, S_idx = -1, -1
            for i, ele in enumerate(ch_names):
                if bi_ch[0] in ele: F_idx = i
                if bi_ch[1] in ele: S_idx = i
            if F_idx !=-1 and S_idx !=-1:
                ch_eeg = eeg_data[F_idx,:] - eeg_data[S_idx,:] #electrode differences
                ch_eeg = (ch_eeg - np.min(ch_eeg))/(np.max(ch_eeg)-np.min(ch_eeg)) #0-1normalization
                ch_lbl = np.zeros(len(ch_eeg))
                #labeling sampling points
                values = ann_dict[key]
                for val in values:
                    st, ed = math.floor(val[0] * down_fq), math.ceil(val[1] * down_fq)
                    ch_lbl[st:ed] = 1

                segs = np.where(np.diff(ch_lbl != 0))[0] + 1
                #segs = np.insert(segs, 0, 0)
                #segs = np.append(segs, len(ch_lbl))
                for p in range(0, len(segs)-1, 2):
                    num = int((segs[p+1]-segs[p])/seg_len)
                    for j in range(num):
                        assert ((segs[p]+(j+1)*seg_len) - (segs[p]+j*seg_len)) == seg_len
                        eeg.append(ch_eeg[segs[p]+j*seg_len:segs[p]+(j+1)*seg_len])
                        lbl.append(ch_lbl[segs[p]+j*seg_len:segs[p]+(j+1)*seg_len])
        
        return eeg, lbl

    def _parse_annotation(self, ann_path):
        montage = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'A1-T3', 'T3-C3', 'C3-CZ',\
                   'CZ-C4', 'C4-T4', 'T4-A2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']
        ann_dict = {}
        with open(ann_path, 'r') as ann_file:
            for line in ann_file.readlines():
                line = line.split(',')
                ch, st, ed, cl =  eval(line[0]), eval(line[1]), eval(line[2]), eval(line[3])
                ch_name = montage[ch]
                if ch_name in ann_dict.keys():
                    ann_dict[ch_name].append((st,ed))
                else:
                    ann_dict[ch_name] = [(st,ed)]

        return ann_dict
    
    def _wavelet_transform(self, eeg):
        fc = pywt.central_frequency('cgau8')
        scales = (2 * fc * self.down_fq) / np.arange(self.down_fq, 0, -1)
        eeg_wt, _ = pywt.cwt(eeg, scales, 'cgau8', 1.0/self.down_fq)
        return eeg_wt
    
def build_dataset(down_fq, seg_len):
    dir = "/data/fjsdata/EEG/JNU-SPSW/files2/tuev_spsw/"
    ids = []
    for _, _, files in os.walk(dir):
        for file in files:
            name = os.path.splitext(file)[0]
            if name not in ids:
                ids.append(name)

    eegs, lbls = [], []
    for id in ids:
        spsw = SPSWInstance(id, down_fq, seg_len)
        eegs.extend(spsw.eeg)
        lbls.extend(spsw.lbl)

    return np.array(eegs), np.array(lbls)

def main():
    eegs, lbls = build_dataset(down_fq=250, seg_len=250)
    print(eegs.shape)
    print(lbls.shape)
    #plot labeling effects
    fig, axes = plt.subplots(1,2, constrained_layout=True,figsize=(12,6))
    for i in range(2):   
        eeg, lbl = eegs[i], lbls[i]
        x = [id for id in range(1, len(lbl)+1)]
        axes[i].plot(x, eeg, color = 'g')
        axes[i].grid(b=True, ls=':')

    fig.savefig('/data/pycode/MedIR/EEG/JNU-SPSW/imgs/tuev_segs.png', dpi=300, bbox_inches='tight') 

if __name__ == "__main__":
    main()
    #nohup python3 Generator.py > /data/tmpexec/tb_log/generator.log 2>&1 &

    
            
    
