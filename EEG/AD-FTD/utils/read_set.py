import mne
import matplotlib.pyplot as plt

raw = mne.io.read_raw_eeglab("/data/fjsdata/EEG/AD-FTD/derivatives/sub-069/eeg/sub-069_task-eyesclosed_eeg.set",preload=False)
#raw = mne.io.read_raw_eeglab("/data/fjsdata/EEG/AD-FTD/sub-058/eeg/sub-058_task-eyesclosed_eeg.set",preload=False)


#np_data = raw.get_data()
sfreq = raw.info['sfreq']
print(sfreq)
ch_names = raw.info['ch_names']
print(ch_names)

#raw.plot(start=5, duration=1)

#layout_from_raw = mne.channels.make_eeg_layout(raw.info)
#layout_from_raw.plot()

#raw.plot_psd()

#data,times=raw[:1,int(sfreq*1):int(sfreq*3)]
#plt.plot(times,data.T)

events_from_annot, event_dict = mne.events_from_annotations(raw)
epochs = mne.Epochs(raw, events=events_from_annot)
print(event_dict)
print(events_from_annot)
print(epochs.get_data().shape)

#raw.plot(events=events_from_annot, start=1, duration=5, color='gray', event_color={1: 'r'})
#plt.savefig('/data/pycode/MedIR/EEG/AD-FTD/imgs/eeg_events_samples.png', bbox_inches='tight')

#epochs = mne.Epochs(raw, events=events_from_annot, event_id=event_dict)
#epochs.plot_image() 
#plt.savefig('/data/pycode/MedIR/EEG/AD-FTD/imgs/eeg_epochs.png', bbox_inches='tight')

#fig = mne.viz.plot_events(events_from_annot, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_dict)
#fig.subplots_adjust(right=0.7)

#plt.savefig('/data/pycode/MedIR/EEG/AD-FTD/imgs/eeg_samples.png', bbox_inches='tight')

#ica = mne.preprocessing.ICA(n_components=2, random_state=97, max_iter=800)
#ica.fit(raw)
#ica.plot_properties(raw)
#plt.savefig('/data/pycode/MedIR/EEG/AD-FTD/imgs/eeg_ica.png', bbox_inches='tight')