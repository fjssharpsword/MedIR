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
        edf_path =  "/data/fjsdata/EEG/JNU-SPSW/files1/" + id + '.edf'
        ann_path =  "/data/fjsdata/EEG/JNU-SPSW/files1/" + id + '.csv'
        raw_np = mne.io.read_raw_edf(edf_path, preload=True)

        #reference lead
        #ch_names = raw_np.info['ch_names'] #electrodes
        #if 'EEG CZ-REF' in ch_names:
        #    raw_np.set_eeg_reference(ref_channels=['EEG CZ-REF'])
        #if 'EEG CZ-LE' in ch_names:
        #    raw_np.set_eeg_reference(ref_channels=['EEG CZ-LE'])

        #filter
        raw_np.filter(l_freq=1, h_freq=70) 
        #raw_np.notch_filter(freqs=50)

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
    
    def _wavelet_transform(self, eeg):
        fc = pywt.central_frequency('cgau8')
        scales = (2 * fc * self.down_fq) / np.arange(self.down_fq, 0, -1)
        eeg_wt, _ = pywt.cwt(eeg, scales, 'cgau8', 1.0/self.down_fq)
        return eeg_wt
    
def build_dataset(down_fq, seg_len):
    dir = "/data/fjsdata/EEG/JNU-SPSW/files1/"
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
        segs = np.where(np.diff(lbl != 0))[0] + 1
        axes[i].plot(segs[0], eeg[segs[0]], marker='^', color='r')
        axes[i].plot(segs[1], eeg[segs[1]], marker='v', color='r')
        axes[i].grid(b=True, ls=':')

    fig.savefig('/data/pycode/MedIR/EEG/SPSW/imgs/eeg_segs.png', dpi=300, bbox_inches='tight') 

if __name__ == "__main__":
    main()
    #nohup python3 Generator.py > /data/tmpexec/tb_log/generator.log 2>&1 &

    
            
    
