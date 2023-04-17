import glob
import numpy as np

from chb_edf_file import ChbEdfFile
from chb_label_wrapper import ChbLabelWrapper

class Patient:
    def __init__(self, id):
        self._id = id
        self._edf_files = list(map(
            lambda filename: ChbEdfFile(filename, self._id),
            glob.glob("/data/fjsdata/EEG/CHB-MIT/files/chb%02d/*.edf" % self._id)
        ))
        self._cumulative_duration = [0]
        
        for file in self._edf_files:#[:-1]:
            self._cumulative_duration.append(int(self._cumulative_duration[-1] + file.get_file_duration()*file.get_sampling_rate()))
        
        self._duration = self._cumulative_duration[-1]#frequency
        
        self._seizure_list = ChbLabelWrapper("/data/fjsdata/EEG/CHB-MIT/files/chb%02d/chb%02d-summary.txt" % (self._id, self._id)).get_seizure_list()
        self._seizure_intervals = []

        for i, file in enumerate(self._seizure_list):
            for seizure in file:
                begin = seizure[0] * self._edf_files[i].get_sampling_rate() + self._cumulative_duration[i]
                end = seizure[1] * self._edf_files[i].get_sampling_rate() + self._cumulative_duration[i]
                self._seizure_intervals.append((int(begin), int(end)))

    def get_channel_names(self):
        #ch_names_com = np.array(self._edf_files[0].get_channel_names())
        ch_names_com = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', \
                        'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8']
        for file in self._edf_files:
            ch_names = np.array(file.get_channel_names())
            ch_names_com = np.intersect1d(ch_names_com, ch_names)
            
        return ch_names_com
    
    def get_sampling_rate(self):
        return self._edf_files[0].get_sampling_rate()

    def get_eeg_data(self, ch_com):
        data_pat = None
        for i, file in enumerate(self._edf_files):
            print ("Reading EEG data from file %s" % file._filename)
            ch_file = file.get_channel_names()
            if not i:
                for j, ch in enumerate(ch_com):
                    if not j:
                        data_file = file.get_channel_data(ch_file.index(ch))
                        data_file = data_file.reshape(-1,1)
                    else:
                        data_ch = file.get_channel_data(ch_file.index(ch))
                        data_ch = data_ch.reshape(-1,1)
                        data_file = np.hstack((data_file,data_ch))
                data_pat = data_file
            else:#align channels
                for j, ch in enumerate(ch_com):
                    if not j:
                        data_file = file.get_channel_data(ch_file.index(ch))
                        data_file = data_file.reshape(-1,1)
                    else:
                        data_ch = file.get_channel_data(ch_file.index(ch))
                        data_ch = data_ch.reshape(-1,1)
                        data_file = np.hstack((data_file,data_ch))
                data_pat = np.vstack((data_pat, data_file))

        return data_pat

    def get_seizures(self):
        return self._seizure_list

    def close_files(self):
        for file in self._edf_files:
            file._file.close()

    def get_seizure_intervals(self):
        return self._seizure_intervals

    def get_seizure_labels(self):
        labels = np.zeros(self._duration)

        for i, interval in enumerate(self._seizure_intervals):
                labels[int(interval[0]):int(interval[1])] = 1

        return labels

    def get_seizure_clips(self):
        clips = []
        data = self.get_eeg_data()
        labels = self.get_seizure_labels()

        for i in range(len(self._seizure_intervals)):
            if not i:
                left = 0
            else:
                left = int((self._seizure_intervals[i-1][1] + self._seizure_intervals[i][0]) / 2)
            if i == len(self._seizure_intervals) - 1:
                right = -1
            else:
                right = int((self._seizure_intervals[i][1] + self._seizure_intervals[i+1][0]) / 2)
            clips.append((data[left:right], labels[left:right]))
        
        return clips
    
if __name__ == "__main__":
    pat = Patient(24)
    data_pat = pat.get_seizure_clips()
    print(data_pat)