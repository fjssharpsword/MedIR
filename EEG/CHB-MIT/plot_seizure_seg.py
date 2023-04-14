import numpy as np
from patient import Patient
import mne
import matplotlib.pyplot as plt

def plot_seizure_segment(patient_id, seconds=2):

    p = Patient(patient_id)
    sp_rate = p.get_sampling_rate()
    win_len = int(sp_rate* seconds)
    ch_com, _ = p.get_channel_names()
    
    #read eeg data
    data_pat = p.get_eeg_data()
    #positive samples
    sei_seg = p._seizure_intervals[0]
    sei_data = data_pat[sei_seg[0]:sei_seg[0]+win_len]
    sei_data = sei_data.transpose(1,0)
    sei_data = sei_data[:2] #two channels

    #negative samples
    non_sei_seg = p._seizure_intervals[1][0] - p._seizure_intervals[0][1]
    non_sei_data = data_pat[non_sei_seg:non_sei_seg+win_len]
    non_sei_data = non_sei_data.transpose(1,0)
    non_sei_data = non_sei_data[:2] #two channels
    
    #https://mne.tools/stable/auto_tutorials/simulation/10_array_objs.html
    #https://blog.51cto.com/u_6811786/4967260
    """
    info = mne.create_info(ch_com[:2].tolist(), sfreq=sp_rate)# Create some dummy metadata
    sei_seg = mne.io.RawArray(sei_data, info)
    sei_seg.plot(show_scrollbars=False, show_scalebars=False)
    plt.savefig('/data/pycode/MedIR/EEG/CHB-MIT/imgs/sei_seg.png', bbox_inches='tight')
    """
    fig, axes = plt.subplots(1,2, constrained_layout=True,figsize=(12,6))

    x = [id for id in range(1, win_len+1)]

    axes[0].plot(x, sei_data[0],color = 'r',label="Seizures")
    axes[0].plot(x, non_sei_data[0],color = 'g',label="Non-seizures")
    axes[0].grid(b=True, ls=':')
    axes[0].legend(loc = "best")
    axes[0].set_title(ch_com[0])

    axes[1].plot(x, sei_data[1],color = 'r',label="Seizures")
    axes[1].plot(x, non_sei_data[1],color = 'g',label="Non-seizures")
    axes[1].grid(b=True, ls=':')
    axes[1].legend(loc = "best")
    axes[1].set_title(ch_com[1])

    fig.savefig('/data/pycode/MedIR/EEG/CHB-MIT/imgs/EEG_Seg.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    plot_seizure_segment(patient_id=1)