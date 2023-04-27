import mne
import matplotlib.pyplot as plt

raw = mne.io.read_raw_eeglab("/data/fjsdata/EEG/AD-FTD/sub-058/eeg/sub-058_task-eyesclosed_eeg.set",preload=False)

#np_data = raw.get_data()
#sfreq = raw.info['sfreq']
#ch_names = raw.info['ch_names']

#raw.plot(start=5, duration=1)

#layout_from_raw = mne.channels.make_eeg_layout(raw.info)
#layout_from_raw.plot()

raw.plot_psd()
plt.savefig('/data/pycode/MedIR/EEG/AD-FTD/imgs/eeg_psd_cn.png', bbox_inches='tight')