import os
import numpy as np
import math
from sklearn.model_selection import KFold
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pywt
from tensorboardX import SummaryWriter
#scikit-learn
from sklearn import svm
from sklearn.model_selection import cross_val_score

PATH_TO_DST_ROOT = '/data/pycode/MedIR/EEG/CHB-MIT/dsts/'

def Train_Eval():

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    #log_writer = SummaryWriter('/data/tmpexec/tb_log')
    
    print('********************Train and validation********************')
    X, y = np.load(PATH_TO_DST_ROOT+'eeg_kfold_2s.npy'), np.load(PATH_TO_DST_ROOT+'lbl_kfold_2s.npy') #time domain

    #X = np.fft.fft(X, axis=1) #Fourier transform, frequence domian

    #X_cA, X_cD = pywt.dwt(X, 'haar', mode='symmetric', axis=1) #wavelet transform, time-frequence domain
    #X = np.concatenate((X_cA, X_cD), axis=1)

    clf = svm.SVC(kernel='linear', C=1, random_state=42)
    X = X[:,:,0]
    scores = cross_val_score(clf, X.reshape(X.shape[0], -1), y, cv=10)

    print('\n Average performance: Accuracy={:.2f}+/-{:.2f}'.format(np.mean(scores)*100, np.std(scores)*100))

def main():
    Train_Eval()

if __name__ == "__main__":
    main()
    #nohup python3 inter_patient_task_ML.py > logs/intra_patient_task.log 2>&1 &