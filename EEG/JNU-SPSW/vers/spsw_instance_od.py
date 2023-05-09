import mne
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

class SPSWInstance:

    def __init__(self, ins_id):

        #parse edf and csv
        edf_path =  "/data/fjsdata/EEG/JNU-SPSW/files1/" + id + '.edf'
        ann_path =  "/data/fjsdata/EEG/JNU-SPSW/files1/" + id + '.csv'
        self.raw_np = mne.io.read_raw_edf(edf_path, preload=True)
        self.ann_df = self._parse_annotation(ann_path)

        #get SPSW instance
        self.spsw_dict = self._init_instance()

    def _init_instance(self):

        eeg_data = self.raw_np.get_data()
        sfreq = self.raw_np.info['sfreq'] #250Hz
        ch_names = self.raw_np.info['ch_names'] #electrodes

        spsw_dict = {}
        for _, row in pd.DataFrame(self.ann_df).iterrows():
            chs, st, ed = row[1], eval(row[2]), eval(row[3])
     
            ch = chs.split("-", 1) #two electrodes
            F_idx, S_idx = -1, -1
            for i, ele in enumerate(ch_names):
                if ch[0] in ele: F_idx = i
                if ch[1] in ele: S_idx = i

            if F_idx !=-1 and S_idx !=-1:
                ch_data = eeg_data[F_idx,:] - eeg_data[S_idx,:]
                st, ed = math.floor(st * sfreq), math.ceil(ed * sfreq)
                if chs in spsw_dict.keys():
                    spsw_dict[chs].append(ch_data[st:ed])
                else:
                    spsw_dict[chs] = [ch_data[st:ed]]

        return spsw_dict

    def _parse_annotation(self, ann_path):

        ann_df =[]
        with open(ann_path, 'r') as ann_file:
            for line in ann_file.readlines()[4:]:
                ann_df.append(line.replace('\n', '').replace('\r', '').replace(' ', '').split(","))

        return ann_df
    
    def get_channel_names(self):
        return self.raw_np.info['ch_names']
    
    def get_sampling_rate(self):
        return self.raw_np.info['sfreq']

if __name__ == "__main__":

    dir = "/data/fjsdata/EEG/JNU-SPSW/files1/"
    ids = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            name = os.path.splitext(file)[0]
            if name not in ids:
                ids.append(name)
    for id in ids:
        spsw = SPSWInstance(id)
        print(spsw.spsw_dict)
