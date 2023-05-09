import mne
import math
import numpy as np
import pandas as pd
import random
import os
import pickle 
import matplotlib.pyplot as plt

def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

class SPSWInstance:
    def __init__(self, id, down_fq=250, seg_len=250*2):

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
        self.eeg, self.lbl = self._parse_EEGWave(raw_np, ann_dict, down_fq, seg_len)

    def _parse_EEGWave(self, raw_np, ann_dict, down_fq, seg_len):

        eeg_data = raw_np.get_data() #numpy data
        ch_names = raw_np.info['ch_names'] #electrodes
        eeg, lbl = [], []
        for key in ann_dict.keys(): 
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
                #seg_lbl = np.split(ch_lbl, segs)
                #seg_data = np.split(ch_eeg, segs)
                for p in range(0, len(segs)-1, 2):
                    if seg_len <= segs[p+1] - segs[p]:
                        num = int((segs[p+1]-segs[p])/seg_len)
                        for j in range(num):
                            assert ((segs[p]+(j+1)*seg_len) - (segs[p]+j*seg_len)) == seg_len
                            eeg.append(ch_eeg[segs[p]+j*seg_len:segs[p]+(j+1)*seg_len])
                            lbl.append(ch_lbl[segs[p]+j*seg_len:segs[p]+(j+1)*seg_len])
                    else: #seg_len > segs[p+1] - segs[p]
                        trun = seg_len - ( segs[p+1] - segs[p] )
                        if trun%2 == 0: #even
                            rem = int(trun/2)
                            if segs[p]-rem>0 and segs[p+1]+rem<len(ch_lbl):
                                assert ((segs[p+1]+rem) - (segs[p]-rem)) == seg_len
                                eeg.append(ch_eeg[segs[p]-rem:segs[p+1]+rem])
                                lbl.append(ch_lbl[segs[p]-rem:segs[p+1]+rem])
                        else: #odd
                            rem = int(trun/2)
                            if segs[p]-rem>0 and segs[p+1]+rem+1<len(ch_lbl):
                                assert ((segs[p+1]+rem+1) - (segs[p]-rem)) == seg_len
                                eeg.append(ch_eeg[segs[p]-rem:segs[p+1]+rem+1])
                                lbl.append(ch_lbl[segs[p]-rem:segs[p+1]+rem+1])

        return eeg, lbl

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

    eegs, lbls = [], []
    for id in ids:
        spsw = SPSWInstance(id)
        eegs.extend(spsw.eeg)
        lbls.extend(spsw.lbl)

    return np.array(eegs), np.array(lbls)

def main():
    eegs, lbls = build_dataset()
    print(eegs.shape)
    print(lbls.shape)

if __name__ == "__main__":
    main()
    #nohup python3 spsw_retrieval.py > /data/tmpexec/tb_log/spsw_retrieval.log 2>&1 &

    
            
    
