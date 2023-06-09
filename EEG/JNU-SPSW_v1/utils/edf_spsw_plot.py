import scipy.io as scio
import mne
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import FastICA
import pywt
import torchaudio
#import librosa
import torch
#https://www.kaggle.com/competitions/seizure-prediction/data
#https://www.sohu.com/a/223132211_740802

def EEG_MelSpectrogram(wave):
    mean, std = -4, 4
    wave_tensor = torch.from_numpy(wave).float()
    to_mel = torchaudio.transforms.MelSpectrogram(n_mels=32, n_fft=128, win_length=32, hop_length=30)
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def main():

    edf_file = '/data/fjsdata/EEG/JNU-SPSW/files1/00002521_s002_t000.edf'
    raw = mne.io.read_raw_edf(edf_file, preload=True)
    sfreq = raw.info['sfreq']
    ch_names = raw.info['ch_names'] #electrodes

    #if  'EEG CZ-REF' in ch_names:
    #    raw.set_eeg_reference(ref_channels=['EEG CZ-REF'])
    #if 'EEG CZ-LE' in ch_names:
    #    raw.set_eeg_reference(ref_channels=['EEG CZ-LE'])
    raw.filter(l_freq=1, h_freq=70) 
    #raw.notch_filter(freqs=50)
    if sfreq!=250:
        raw.resample(250, npad="auto")
        sfreq = 250

    #lbl_file = '/data/fjsdata/EEG/JNU-SPSW/files1/00013145_s004_t006.csv'
    #lbl_df = pd.read_csv(lbl_file, sep=',')
    # Using readlines()
    lbl_file = open('/data/fjsdata/EEG/JNU-SPSW/files1/00002521_s002_t000.csv', 'r')
    lbl_df = []
    for line in lbl_file.readlines()[4:]:
        lbl_df.append(line.replace('\n', '').replace('\r', '').replace(' ', '').split(","))
    
    eeg_data = raw.get_data()
    spsw_span = []
    for idx, row in pd.DataFrame(lbl_df).iterrows():
        chs, st, ed = row[1], eval(row[2]), eval(row[3])

        #st, ed = math.floor(st * sfreq), math.ceil(ed * sfreq)
        #spsw_span.append( eeg_data[0,st-100:ed+100] )
        if chs != 'T6-O2': continue
        ch = chs.split("-", 1) #two electrodes
        F_idx, S_idx = -1, -1
        for i, ele in enumerate(ch_names):
            if ch[0] in ele: F_idx = i
            if ch[1] in ele: S_idx = i

        if F_idx !=-1 and S_idx !=-1:
            ch_data = eeg_data[F_idx,:] - eeg_data[S_idx,:]
            st, ed = math.floor(st * sfreq), math.ceil(ed * sfreq)
            spsw_span.append(ch_data[st-100:ed+100])

            #fc = pywt.central_frequency('cgau8')
            #scales = (2 * fc * 250) / np.arange(250, 0, -1)
            #X = ch_data[st-250:ed+250]
            #X, _ = pywt.cwt(X, scales, 'cgau8', 1.0/250)
            #print(X.shape)
        
    #plot
    fig, axes = plt.subplots(1,2, constrained_layout=True,figsize=(12,6))
    for i in range(2):
        span = spsw_span[i+2]
        span = EEG_MelSpectrogram(span)
        x = [id for id in range(1, len(span)+1)]
        axes[i].plot(x, span, color = 'b', label='T6-O2')
        axes[i].plot(x[100], span[100], marker='^', color='r')
        axes[i].plot(x[len(x)-100], span[len(x)-100], marker='v', color='r')
        axes[i].grid(b=True, ls=':')
        axes[i].legend(loc='best')

    fig.savefig('/data/pycode/MedIR/EEG/JNU-SPSW/imgs/eeg_time_spsw.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()