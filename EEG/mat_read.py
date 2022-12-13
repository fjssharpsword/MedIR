import scipy.io as scio
import mne
import numpy as np
import matplotlib.pyplot as plt

#https://www.kaggle.com/competitions/seizure-prediction/data
def main():

    dataFile = '/data/pycode/MedIR/EEG/data/Dog_2_interictal_segment_0001.mat'
    data = scio.loadmat(dataFile)

    print (data.keys())#['__header__', '__version__', '__globals__', 'interictal_segment_1']
    data = data['interictal_segment_1'][0]
    samples = data['data'][0]
    length_sec = data['data_length_sec'][0]
    freq = data['sampling_frequency'][0]
    seq = data['sequence'][0]
    
    ch_names = ['1', '2','3','4','5','6', '7','8','9','10','11', '12','13','14','15','16'] #通道名称
    sfreq = freq
    info = mne.create_info(ch_names, sfreq) 
    raw = mne.io.RawArray(samples, info) 

    raw.plot()
    plt.savefig('/data/pycode/MedIR/EEG/imgs/eeg_mat.png', bbox_inches='tight')
    print('shape：', raw.get_data().shape)
    print('channel：', raw.info.get('nchan'))


if __name__ == "__main__":
    main()