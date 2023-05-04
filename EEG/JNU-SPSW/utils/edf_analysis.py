import scipy.io as scio
import mne
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#https://www.kaggle.com/competitions/seizure-prediction/data
def main():

    edf_file = '/data/tmpexec/tb_log/eeg-case/00000006_s004_t000.edf'
    raw = mne.io.read_raw_edf(edf_file)

    #events_from_annot, event_dict = mne.events_from_annotations(raw)
    #print(event_dict)
    #print(events_from_annot)  

    lbl_file = '/data/tmpexec/tb_log/eeg-case/00000006_s004_t000.csv'
    lbl_df = pd.read_csv(lbl_file, sep=',')

    sfreq = raw.info['sfreq']
    ch_names = raw.info['ch_names'] #electrodes
    eeg_data = raw.get_data()

    spsw_span = []
    for idx, row in lbl_df.iterrows():
        chs, st, ed = row[1], row[2], row[3]

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

    fig.savefig('/data/pycode/MedIR/EEG/JNU-SPSW/imgs/EEG_Seg.png', dpi=300, bbox_inches='tight')

    #raw.plot(start=5, duration=1)#time domain
    #raw.plot_psd() #frequency domain
    #data,times=raw[:2,spsw_span[0][0]:spsw_span[0][1]]
    #plt.plot(times,data.T)
    #plt.savefig('/data/pycode/MedIR/EEG/JNU-SPSW/imgs/eeg_vis.png', bbox_inches='tight')


if __name__ == "__main__":
    main()