import scipy.io as scio
import mne
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import FastICA

#https://www.kaggle.com/competitions/seizure-prediction/data
#https://www.sohu.com/a/223132211_740802

def main():

    edf_file = '/data/fjsdata/EEG/JNU-SPSW/files1/00000002_s005_t000.edf'
    raw = mne.io.read_raw_edf(edf_file, preload=True)
    sfreq = raw.info['sfreq']
    ch_names = raw.info['ch_names'] #electrodes

    #if  'EEG CZ-REF' in ch_names:
    #    raw.set_eeg_reference(ref_channels=['EEG CZ-REF'])
    #if 'EEG CZ-LE' in ch_names:
    #    raw.set_eeg_reference(ref_channels=['EEG CZ-LE'])

    #lbl_file = '/data/fjsdata/EEG/JNU-SPSW/files1/00013145_s004_t006.csv'
    #lbl_df = pd.read_csv(lbl_file, sep=',')
    # Using readlines()
    lbl_file = open('/data/fjsdata/EEG/JNU-SPSW/files1/00000002_s005_t000.csv', 'r')
    lbl_df = []
    for line in lbl_file.readlines()[4:]:
        lbl_df.append(line.replace('\n', '').replace('\r', '').replace(' ', '').split(","))
    
    eeg_data = raw.get_data()
    spsw_span = []
    for idx, row in pd.DataFrame(lbl_df).iterrows():
        chs, st, ed = row[1], eval(row[2]), eval(row[3])

        #st, ed = math.floor(st * sfreq), math.ceil(ed * sfreq)
        #spsw_span.append( eeg_data[0,st-100:ed+100] )
        
        ch = chs.split("-", 1) #two electrodes
        F_idx, S_idx = -1, -1
        for i, ele in enumerate(ch_names):
            if ch[0] in ele: F_idx = i
            if ch[1] in ele: S_idx = i

        if F_idx !=-1 and S_idx !=-1:
            ch_data = eeg_data[F_idx,:] - eeg_data[S_idx,:]
            st, ed = math.floor(st * sfreq), math.ceil(ed * sfreq)
            spsw_span.append( ch_data[st-100:ed+100] )
        
    #plot
    fig, axes = plt.subplots(1,2, constrained_layout=True,figsize=(12,6))

    x = [id for id in range(1, len(spsw_span[0])+1)]
    axes[0].plot(x, spsw_span[0],color = 'b')
    axes[0].plot(100, spsw_span[0][100], marker='^', color='r')
    axes[0].plot(len(spsw_span[0])-100, spsw_span[0][len(spsw_span[0])-100], marker='v', color='r')
    axes[0].grid(b=True, ls=':')

    x = [id for id in range(1, len(spsw_span[1])+1)]
    axes[1].plot(x, spsw_span[1],color = 'b')
    axes[1].plot(100, spsw_span[1][100], marker='^', color='r')
    axes[1].plot(len(spsw_span[1])-100, spsw_span[1][len(spsw_span[1])-100], marker='v', color='r')
    axes[1].grid(b=True, ls=':')

    fig.savefig('/data/pycode/MedIR/EEG/JNU-SPSW/imgs/eeg_spsw_seg.png', dpi=300, bbox_inches='tight')

    """
    raw.plot(start=5, duration=1)#time domain
    raw.plot_psd() #frequency domain
    data,times=raw[:2,spsw_span[0][0]:spsw_span[0][1]]
    plt.plot(times,data.T)
    plt.savefig('/data/pycode/MedIR/EEG/JNU-SPSW/imgs/eeg_vis.png', bbox_inches='tight')
    print(raw.info['bads'])
    raw_ref = mne.add_reference_channels(raw, 'EEG CZ-REF')
    eeg_data = raw_ref.get_data()
    print(eeg_data.shape)
    events_from_annot, event_dict = mne.events_from_annotations(raw)
    print(event_dict)
    print(events_from_annot)  
    event_dict = mne.find_events(raw, stim_channel='STI 014')
    print(event_dict)
    layout_from_raw = mne.channels.make_eeg_layout(raw.info)
    layout_from_raw.plot()
    #filt_raw = raw.copy().filter(l_freq=1., h_freq=None)
    #ica_model = mne.preprocessing.ICA(n_components=18).fit(raw)
    #ica_raw = ica_model.get_components()
    ica_model = FastICA(n_components=18, random_state=0, whiten='unit-variance')
    raw_ica = ica_model.fit_transform(raw.get_data().transpose(1,0))
    """

if __name__ == "__main__":
    main()