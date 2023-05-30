import mne
import math
import numpy as np
import pandas as pd
import random
import os
import pickle 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from nets.sa_unet import build_unet, DiceLoss
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import confusion_matrix

def build_dataset():
    PATH_TO_DST_ROOT = '/data/pycode/MedIR/EEG/TUSZ/dsts/'
    X, y = np.load(PATH_TO_DST_ROOT+'tusz_tr_eeg_1s.npy'), np.load(PATH_TO_DST_ROOT+'tusz_tr_lbl_1s.npy')
    print('\r Sample number: {}'.format(len(y)))

    dataset = TensorDataset(torch.FloatTensor(X).unsqueeze(1), torch.LongTensor(y))
    dataloader = DataLoader(dataset=dataset, batch_size=8192,shuffle=False, num_workers=2, pin_memory=True)
    return dataloader

def main():
    #loading model
    CKPT_PATH = '/data/pycode/MedIR/EEG/JNU-SPSW/ckpts/sa_unet_t.pkl'
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    model = build_unet(in_ch=1, n_classes=1).to(device)
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained model: "+CKPT_PATH)
    model.eval()#turn to evaluation mode

    #laoding dataset
    dataloader = build_dataset()

    gt_lbl = torch.FloatTensor()
    pr_lbl = torch.FloatTensor()
    te_acc = 0.0
    for eegs, lbls in dataloader:
        var_eeg = eegs.to(device)
        var_lbl = lbls.to(device)
        var_out = model(var_eeg)

        var_out = torch.where(var_out>0.5, 1, 0)
        var_out = var_out.view(var_out.size(0), -1)
        var_prd = torch.div(var_out.sum(1), var_out.size(1))
        var_prd = torch.where(var_prd>0.18, 1, 0)
        pr_lbl = torch.cat((pr_lbl, var_prd.cpu()), 0)
        gt_lbl = torch.cat((gt_lbl, lbls), 0)
        
        te_acc += (var_prd == var_lbl).sum().item()

    tn, fp, fn, tp = confusion_matrix(gt_lbl.numpy(), pr_lbl.numpy()).ravel()
    te_sen = tp /(tp+fn)
    te_spe = tn /(tn+fp)
    te_acc = te_acc/len(gt_lbl)
    print('\n Accuracy={:.2f}, Sensitivity={:.2f}, Specificity={:.2f}'.format(te_acc*100, te_sen*100, te_spe*100))

if __name__ == "__main__":
    main()
    #nohup python3 evaluator.py > /data/tmpexec/tb_log/evaluator.log 2>&1 &

    
            
    
