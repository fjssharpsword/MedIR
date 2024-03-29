import mne
import math
import numpy as np
import pandas as pd
import random
import os
import pickle 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from nets.sa_unet import build_unet, DiceLoss
#from nets.utime import build_unet, DiceLoss

def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

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

        #get SPSW instance
        ann_dict = self._parse_annotation(ann_path)
        self.eeg, self.lbl = self._parse_EEGWave(raw_np, ann_dict, down_fq, seg_len)

    def _parse_EEGWave(self, raw_np, ann_dict, down_fq, seg_len):

        eeg_data = raw_np.get_data() #numpy data
        ch_names = raw_np.info['ch_names'] #electrodes
        eeg, lbl = [], []
        for key in ann_dict.keys(): 

            #screening out specified channels
            if key not in ['T6-O2']: continue #['FP2-F8']

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

    #processing dataset
    eegs, lbls = build_dataset(down_fq=250, seg_len=250)
    print('\r Dataset scale')
    print(eegs.shape)
    print(lbls.shape)

    #loading model
    CKPT_PATH = '/data/pycode/MedIR/EEG/JNU-SPSW/ckpts/unet_t_T6-O2.pkl'
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    model = build_unet(in_ch=1, n_classes=1).to(device)
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained model: "+CKPT_PATH)
    model.eval()#turn to evaluation mode

    #plot labeling effects
    fig, axes = plt.subplots(1,2, constrained_layout=True,figsize=(12,6))
    ax_idx = 0

    for i in range(len(lbls)):
        if ax_idx > 1: break   

        eeg, gt_lbl = eegs[i], lbls[i]
        #prediction
        X = torch.FloatTensor(eeg).unsqueeze(0).unsqueeze(0)
        pd_lbl = model(X.to(device))
        pd_lbl = torch.where(torch.flatten(pd_lbl)>0.5, 1, 0)
        pd_lbl = pd_lbl.cpu().numpy()
        #show
        if dice_coef(pd_lbl, gt_lbl) > 0.95:
            pt = [id for id in range(1, len(gt_lbl)+1)]
            axes[ax_idx].plot(pt, eeg, color = 'g')
            for j in range(len(gt_lbl)):
                if int(gt_lbl[j]) == 1: 
                    axes[ax_idx].scatter(pt[j], eeg[j], color='b', marker='v', alpha=0.5)
            for j in range(len(pd_lbl)):
                if int(pd_lbl[j]) == 1: 
                    axes[ax_idx].scatter(pt[j], eeg[j], color='r', marker='^', alpha=0.5)
            axes[ax_idx].grid(b=True, ls=':')
            ax_idx = ax_idx+1

    fig.savefig('/data/pycode/MedIR/EEG/JNU-SPSW/imgs/eeg_gt_pred_2.png', dpi=300, bbox_inches='tight') 

if __name__ == "__main__":
    main()
    #nohup python3 plotter.py > /data/tmpexec/tb_log/plotter.log 2>&1 &

    
            
    
