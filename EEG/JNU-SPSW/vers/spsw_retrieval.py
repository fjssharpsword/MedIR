import mne
import math
import numpy as np
import pandas as pd
import random
import os
import pickle 
import matplotlib.pyplot as plt

class SPSWInstance:
    def __init__(self, id, down_fq=250, mode='tr'):

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
        if mode == 'tr':
            self.eeg_dict = self._parse_trainID(raw_np, ann_dict, down_fq)
        if mode == 'te':
            self.eeg_dict = self._parse_testID(raw_np, ann_dict, down_fq)

    def _parse_trainID(self, raw_np, ann_dict, down_fq):

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
                ch_eeg = (ch_eeg - np.min(ch_eeg))/(np.max(ch_eeg)-np.min(ch_eeg)) #0-1normalization
                ch_lbl = np.zeros(len(ch_eeg))
                #positive waves
                values = ann_dict[key]
                for val in values:
                    st, ed = math.floor(val[0] * down_fq), math.ceil(val[1] * down_fq)
                    ch_lbl[st:ed] = 1
                    if key in eeg_dict.keys():
                        eeg_dict[key].append(ch_eeg[st:ed])
                    else:
                        eeg_dict[key] = [ch_eeg[st:ed]]
                #negative waves
                #segs = np.where(np.diff(ch_lbl != 0))[0] + 1
                #ch_lbl = np.split(ch_lbl, segs)

        return eeg_dict

    def _parse_testID(self, raw_np, ann_dict, down_fq):
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
                ch_eeg = (ch_eeg - np.min(ch_eeg))/(np.max(ch_eeg)-np.min(ch_eeg)) #0-1normalization
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

    random.shuffle(ids)
    tr_ids = ids[:int(len(ids)*0.8)]
    te_ids = ids[int(len(ids)*0.8):]

    eeg_dict = {}
    for id in tr_ids:
        spsw = SPSWInstance(id, mode='tr')
        for key in spsw.eeg_dict.keys():
            if key in eeg_dict.keys():
                eeg_dict[key].extend(spsw.eeg_dict[key])
            else:
                eeg_dict[key] = spsw.eeg_dict[key]

    dir = "/data/pycode/MedIR/EEG/JNU-SPSW/dsts/"
    with open(dir+'eeg_wr_db.pkl', 'wb') as f:
        pickle.dump(eeg_dict, f)

    return te_ids

def vis_sequence():
    dir = "/data/pycode/MedIR/EEG/JNU-SPSW/dsts/"
    with open(dir+'eeg_wr_db.pkl', 'rb') as f:
        eeg_dict = pickle.load(f)

    fig, axes = plt.subplots(1,2, constrained_layout=True,figsize=(12,6))
    for i in range(2):   
        key = list(eeg_dict.keys())[i+2]  
        vals = eeg_dict[key]
    
        x = [id for id in range(1, len(vals[i])+1)]
        axes[i].plot(x, vals[i], color = 'g', label=key)
        axes[i].legend(loc = "best")
        axes[i].grid(b=True, ls=':')

    fig.savefig('/data/pycode/MedIR/EEG/JNU-SPSW/imgs/eeg_time_seq.png', dpi=300, bbox_inches='tight') 


def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def eval_dataset(te_ids, threshold=0.5):

    #load SPSW database
    dir = "/data/pycode/MedIR/EEG/JNU-SPSW/dsts/"
    with open(dir+'eeg_wr_db.pkl', 'rb') as f:
        tr_dict = pickle.load(f)

    key_dice = {}
    for id in te_ids:
        spsw = SPSWInstance(id, mode='te')
        te_dict = spsw.eeg_dict
        for key in te_dict.keys():
            tr_spsws = tr_dict[key]
            te_eeg, te_lbl = te_dict[key][0], te_dict[key][1]
            pr_lbl = np.zeros(len(te_eeg))#initiatizing predicted labels
            for tr_spsw in tr_spsws:
                num = int(len(te_eeg)/len(tr_spsw))
                for i in range(num):
                    te_spsw = te_eeg[i:len(tr_spsw)+i]
                    dist = np.linalg.norm(tr_spsw - te_spsw) #Euclidean distance
                    #dist = tr_spsw.dot(te_spsw) / (np.linalg.norm(tr_spsw) * np.linalg.norm(te_spsw)) #cosine distance 
                    if dist < threshold: #similarity
                        pr_lbl[i:len(tr_spsw)+i]=1
            #lead_acc = (pr_lbl == te_lbl).sum()/len(te_lbl)
            lead_dice = dice_coef(te_lbl, pr_lbl)
            print('\n ID_{} Key_{}: Dice={:.6f}'.format(id, key, lead_dice))
            if key in key_dice.keys():
                key_dice[key].append(lead_dice)
            else:
                key_dice[key] = [lead_dice]

    return key_dice
        
def main():
    #10-fold cross-validation
    key_dice = {}
    for epoch in range(10):
        te_ids = build_dataset()
        dice_dict = eval_dataset(te_ids)

        for key in dice_dict.keys():
            if key in key_dice.keys():
                key_dice[key].append(np.mean(dice_dict[key]))
            else:
                key_dice[key] = np.mean(dice_dict[key])
    for key in key_dice.keys():
        print('\n {} performance: Dice={:.2f}+/-{:.2f}'.format(key, np.mean(key_dice[key])*100, np.std(key_dice[key])*100))

    #vis_sequence()

if __name__ == "__main__":
    main()
    #nohup python3 spsw_retrieval.py > /data/tmpexec/tb_log/spsw_retrieval.log 2>&1 &

    
            
    
