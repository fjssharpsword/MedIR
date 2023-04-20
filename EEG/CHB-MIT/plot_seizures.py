import numpy as np
import pandas as pd
from inter_patient import Patient
import mne
import matplotlib.pyplot as plt
import pywt

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

def plot_seizure_frequence(patient_id=1, seconds=2):
    p = Patient(patient_id)
    sp_rate = p.get_sampling_rate()
    win_len = int(sp_rate* seconds)
    ch_com = p.get_channel_names()
    
    #read eeg data
    data_pat = p.get_eeg_data(ch_com)
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

    fig, axes = plt.subplots(2,2, constrained_layout=True,figsize=(16,12))

    x = [id for id in range(1, win_len+1)]

    axes[0,0].plot(x, sei_data[0],color = 'r',label="Seizures")
    axes[0,0].plot(x, non_sei_data[0],color = 'g',label="Non-seizures")
    axes[0,0].grid(b=True, ls=':')
    axes[0,0].legend(loc = "best")
    axes[0,0].set_title('Time domain:' + ch_com[0])
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Amplitude')

    axes[0,1].plot(x, sei_data[1],color = 'r',label="Seizures")
    axes[0,1].plot(x, non_sei_data[1],color = 'g',label="Non-seizures")
    axes[0,1].grid(b=True, ls=':')
    axes[0,1].legend(loc = "best")
    axes[0,1].set_title('Time domain:' + ch_com[1])
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Amplitude')

    
    #axes[1,0].plot(x, np.abs(np.fft.fft(sei_data[0])), color = 'r',label="Seizures")
    #axes[1,0].plot(x, np.abs(np.fft.fft(non_sei_data[0])), color = 'g',label="Non-seizures")
    freq = np.fft.fftfreq(n=len(x), d=1.0)
    axes[1,0].stem(freq, np.abs(np.fft.fft(sei_data[0])), 'r', markerfmt=" ", basefmt="-b", label="Seizures")
    axes[1,0].stem(freq, np.abs(np.fft.fft(non_sei_data[0])), 'g', markerfmt=" ", basefmt="-r", label="Non-seizures")
    axes[1,0].grid(b=True, ls=':')
    axes[1,0].legend(loc = "best")
    axes[1,0].set_title('Frequency domain:' + ch_com[0])
    axes[1,0].set_xlabel('Freq (Hz)')
    axes[1,0].set_ylabel('FFT Amplitude |X(freq)|')

    #axes[1,1].plot(x, np.abs(np.fft.fft(sei_data[1])),color = 'r',label="Seizures")
    #axes[1,1].plot(x, np.abs(np.fft.fft(non_sei_data[1])), color = 'g',label="Non-seizures")
    axes[1,1].stem(freq, np.abs(np.fft.fft(sei_data[1])), 'r', markerfmt=" ", basefmt="-b", label="Seizures")
    axes[1,1].stem(freq, np.abs(np.fft.fft(non_sei_data[1])), 'g', markerfmt=" ", basefmt="-r", label="Non-seizures")
    axes[1,1].grid(b=True, ls=':')
    axes[1,1].legend(loc = "best")
    axes[1,1].set_title('Frequency domain:' + ch_com[1])
    axes[1,1].set_xlabel('Freq (Hz)')
    axes[1,1].set_ylabel('FFT Amplitude |X(freq)|')

    fig.savefig('/data/pycode/MedIR/EEG/CHB-MIT/imgs/EEG_Freq.png', dpi=300, bbox_inches='tight')

def plot_seizure_specgram(patient_id=1, seconds=2):
    p = Patient(patient_id)
    sp_rate = p.get_sampling_rate()
    win_len = int(sp_rate* seconds)
    ch_com = p.get_channel_names()
    
    #read eeg data
    data_pat = p.get_eeg_data(ch_com)
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

    fig, axes = plt.subplots(2,2, constrained_layout=True,figsize=(16,12))

    x = [id for id in range(1, win_len+1)]

    Pxx, freqs, bins, im = axes[0,0].specgram(sei_data[0])
    axes[0,0].set_title('Seizures:' + ch_com[0])

    Pxx, freqs, bins, im = axes[0,1].specgram(non_sei_data[0])
    axes[0,1].set_title('Non seizures:' + ch_com[0])

    Pxx, freqs, bins, im = axes[1,0].specgram(sei_data[1])
    axes[1,0].set_title('Seizures:' + ch_com[1])

    Pxx, freqs, bins, im = axes[1,1].specgram(non_sei_data[1])
    axes[1,1].set_title('Non seizures:' + ch_com[1])

    fig.savefig('/data/pycode/MedIR/EEG/CHB-MIT/imgs/EEG_Specgram.png', dpi=300, bbox_inches='tight')

def plot_seizure_wavelet(patient_id=1, seconds=2):
    p = Patient(patient_id)
    sp_rate = p.get_sampling_rate()
    win_len = int(sp_rate* seconds)
    ch_com = p.get_channel_names()
    
    #read eeg data
    data_pat = p.get_eeg_data(ch_com)
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

    fig, axes = plt.subplots(2,2, constrained_layout=True,figsize=(16,12))

    x = [id for id in range(1, win_len+1)]

    wavename = 'cgau8'
    cwtcoff, freq = pywt.cwt(sei_data[0], np.arange(1,win_len+1), wavename)
    axes[0,0].contourf(x, freq, abs(cwtcoff))
    axes[0,0].set_title('Seizures:' + ch_com[0])

    cwtcoff, freq = pywt.cwt(non_sei_data[0], np.arange(1,win_len+1), wavename)
    axes[0,1].contourf(x, freq, abs(cwtcoff))
    axes[0,1].set_title('Non seizures:' + ch_com[0])

    cwtcoff, freq = pywt.cwt(sei_data[1], np.arange(1,win_len+1), wavename)
    axes[1,0].contourf(x, freq, abs(cwtcoff))
    axes[1,0].set_title('Seizures:' + ch_com[1])

    cwtcoff, freq = pywt.cwt(non_sei_data[1], np.arange(1,win_len+1), wavename)
    axes[1,1].contourf(x, freq, abs(cwtcoff))
    axes[1,1].set_title('Non seizures:' + ch_com[1])

    fig.savefig('/data/pycode/MedIR/EEG/CHB-MIT/imgs/EEG_Wavelet.png', dpi=300, bbox_inches='tight')

def plot_seizure_box_channel(patient_id=1, seconds=2):
    p = Patient(patient_id)
    sp_rate = p.get_sampling_rate()
    win_len = int(sp_rate* seconds)
    ch_com = p.get_channel_names()
    
    #read eeg data
    data_pat = p.get_eeg_data(ch_com)
    #positive samples
    sei_seg = p._seizure_intervals[0]
    sei_data = data_pat[sei_seg[0]:sei_seg[0]+win_len]
    sei_data = sei_data.transpose(1,0)
    #sei_data = sei_data[:2] #two channels

    #negative samples
    non_sei_seg = p._seizure_intervals[1][0] - p._seizure_intervals[0][1]
    non_sei_data = data_pat[non_sei_seg:non_sei_seg+win_len]
    non_sei_data = non_sei_data.transpose(1,0)
    #non_sei_data = non_sei_data[:2] #two channels

    fig, axes = plt.subplots(1,2, constrained_layout=True,figsize=(16,8))

    x = [id for id in range(1, win_len+1)]

    sei_data=[sei_data[i] for i in range(0,len(ch_com))]
    axes[0].boxplot(sei_data, labels = ch_com)
    axes[0].set_title('Seizures:')

    non_sei_data=[non_sei_data[i] for i in range(0,len(ch_com))]
    axes[1].boxplot(non_sei_data, labels = ch_com)
    axes[1].set_title('Non seizures:')

    fig.savefig('/data/pycode/MedIR/EEG/CHB-MIT/imgs/EEG_Box_channel.png', dpi=300, bbox_inches='tight')

def plot_seizure_median(patient_id=1, seconds=2):

    p = Patient(patient_id)
    sp_rate = p.get_sampling_rate()
    win_len = int(sp_rate* seconds)
    ch_com = p.get_channel_names()
    
    #read eeg data
    data_pat = p.get_eeg_data(ch_com)
    #positive samples
    sei_data= []
    for sei_seg in p._seizure_intervals:
        num = int((sei_seg[1]-sei_seg[0])/win_len)
        pos = sei_seg[0]
        for i in range(num):
            pos = sei_seg[0]+i*win_len
            sei_data.append(data_pat[pos:pos+win_len])

    #negative samples
    non_sei_data = []
    for i in range(len(p._seizure_intervals)-1):
        if len(non_sei_data) >= len(sei_data): 
            break
        num = int((p._seizure_intervals[i+1][0]-p._seizure_intervals[i][1])/win_len)
        for j in range(num):
            pos = p._seizure_intervals[i][1]+j*win_len
            non_sei_data.append(data_pat[pos:pos+win_len])
    non_sei_data = non_sei_data[:len(sei_data)]

    sei_data = np.array(sei_data).transpose(1,0,2)
    sei_data = sei_data[:2] #two channels
    sei_data = np.median(sei_data, axis=2)
    non_sei_data = np.array(non_sei_data).transpose(1,0,2)
    non_sei_data = non_sei_data[:2] #two channels
    non_sei_data = np.median(non_sei_data, axis=2)

    fig, axes = plt.subplots(1,2, constrained_layout=True,figsize=(16,8))

    x = [id for id in range(1, len(sei_data[0])+1)]

    axes[0].scatter(x, sei_data[0], color = 'r',label="Seizures")
    axes[0].scatter(x, non_sei_data[0], color = 'g',label="Non-seizures")
    axes[0].grid(b=True, ls=':')
    axes[0].legend(loc = "best")
    axes[0].set_title(ch_com[0])

    axes[1].scatter(x, sei_data[1], color = 'r',label="Seizures")
    axes[1].scatter(x, non_sei_data[1], color = 'g',label="Non-seizures")
    axes[1].grid(b=True, ls=':')
    axes[1].legend(loc = "best")
    axes[1].set_title(ch_com[1])

    fig.savefig('/data/pycode/MedIR/EEG/CHB-MIT/imgs/EEG_Box_median.png', dpi=300, bbox_inches='tight')

def plot_wave_median(patient_id=1, seconds=2):

    p = Patient(patient_id)
    ch_com = p.get_channel_names()
    samples = p.get_eeg_data(ch_com)
    #samples = samples.transpose(1,0)
    labels = p.get_seizure_labels()
    sp_rate = p.get_sampling_rate()
    win_len = int(sp_rate* seconds)
    
    first_seizure = p._seizure_intervals[0]
    #last_seizure = p._seizure_intervals[1]
    st_pos = first_seizure[0] - win_len*2 
    end_pos = first_seizure[1] + win_len*2

    segs = samples[st_pos:st_pos+win_len]
    lbls = [0]
    meds = [np.median(segs, axis=0)]
    for i in range(st_pos+win_len, end_pos):
        segs = np.delete(segs, 0, axis=0) #delete the first point
        segs = np.append(segs, samples[i].reshape(1,-1), axis=0) #insert a point
        lbls.append(labels[i])
        meds.append(np.median(segs, axis=0))

    fig, ax = plt.subplots(1,1, constrained_layout=True,figsize=(12,8))

    x = [id for id in range(1, len(lbls)+1)]

    meds = np.array(meds).transpose(1,0)
    meds = meds[:2] #two channels

    ax.plot(x, meds[0], color = 'b', label=ch_com[0])
    ax.plot(x, meds[1], color = 'g', label=ch_com[1])
    v_t = 0
    for a, b, c in zip(x, lbls, meds[0]):
        if b != v_t:
            ax.plot(a, c, color='r', marker='o')
            v_t = b
    #ax.set_xticklabels(lbls)
    ax.grid(b=True, ls=':')
    ax.legend(loc = "best")

    fig.savefig('/data/pycode/MedIR/EEG/CHB-MIT/imgs/EEG_Sample_Median.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    #plot_seizure_segment(patient_id=1)
    #plot_seizure_frequence(patient_id=1)
    #plot_seizure_specgram(patient_id=1)
    #plot_seizure_wavelet(patient_id=1)
    #plot_seizure_box_channel(patient_id=1)
    #plot_seizure_median(patient_id=1)
    plot_wave_median(patient_id=1)