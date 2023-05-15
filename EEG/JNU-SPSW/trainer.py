import os
import numpy as np
import math
from sklearn.model_selection import KFold
import torch
from torchvision import transforms
import torch.nn as nn
import torchvision
import random
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import confusion_matrix
import pywt
from tensorboardX import SummaryWriter
#self-defined
from nets.ConvUNet import build_unet, DiceLoss
from dsts.Generator import build_dataset, dice_coef
from nets.LSTMUNet import LSTMSeg

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    tr_loss = []
    model.train()
    for eegs, lbls in dataloader:
        var_eeg = eegs.to(device)
        var_lbl = lbls.to(device)
        optimizer.zero_grad()
        var_out = model(var_eeg)
        loss = loss_fn(var_out, var_lbl)
        loss.backward()
        optimizer.step()
        tr_loss.append(loss.item())

    tr_loss = np.mean(tr_loss)
    return tr_loss

def eval_epoch(model, dataloader, loss_fn, device):
    te_loss = []
    pr_lbl = torch.FloatTensor()
    gt_lbl = torch.FloatTensor()
    model.eval()
    for eegs, lbls in dataloader:
        var_eeg = eegs.to(device)
        var_lbl = lbls.to(device)
        var_out = model(var_eeg)
        loss = loss_fn(var_out, var_lbl)
        te_loss.append(loss.item())

        gt_lbl = torch.cat((gt_lbl, lbls), 0)
        var_out = torch.where(var_out>0.5, 1, 0)
        pr_lbl = torch.cat((pr_lbl, var_out.cpu()), 0)

    te_loss = np.mean(te_loss)
    te_coef = dice_coef(gt_lbl, pr_lbl)

    return te_coef

def Train_Eval():

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    #log_writer = SummaryWriter('/data/tmpexec/tb_log')
    
    print('********************Train and validation********************')
    X, y = build_dataset(down_fq=250, seg_len=250) #time domain

    #X = np.fft.fft(X, axis=1) #Fourier transform, frequence domain

    #X_cA, X_cD = pywt.dwt(X, 'haar', mode='symmetric', axis=1) #wavelet transform, time-frequence domain
    #X = np.concatenate((X_cA, X_cD), axis=1)

    print('\r Sample number: {}'.format(len(y)))
    dataset = TensorDataset(torch.FloatTensor(X).unsqueeze(1), torch.LongTensor(y))
    kf_set = KFold(n_splits=10, shuffle=True).split(X, y)

    dice_list = []
    for f_id, (tr_idx, te_idx) in enumerate(kf_set):
        print('\r Fold {} train and validation.'.format(f_id + 1))
        #dataset
        te_sampler = SubsetRandomSampler(te_idx)
        te_dataloader = DataLoader(dataset, batch_size = 512, sampler=te_sampler)
        tr_sampler = SubsetRandomSampler(tr_idx)
        tr_dataloader = DataLoader(dataset, batch_size = 512, sampler=tr_sampler)
        
        #model 
        #model = build_unet(in_ch=1, n_classes=1).to(device)  #conv-unet
        model = LSTMSeg(num_electrodes = 1, hid_channels=64, num_classes=1).to(device) #conv-unet
        
        optimizer_model = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
        criterion = DiceLoss() #nn.CrossEntropyLoss() 
        #cross-validation
        best_dice = 0.0
        for epoch in range(100):
            tr_loss = train_epoch(model, tr_dataloader, criterion, optimizer_model, device)
            lr_scheduler_model.step()  #about lr and gamma
            te_coef = eval_epoch(model, te_dataloader, criterion, device)

            #log_writer.add_scalars('EEG/CHB-MIT/Loss', {'Train':tr_loss, 'Test':te_loss}, epoch+1)
            print('\r Train Epoch_{}: DiceLoss={:.4f}'.format(epoch+1, tr_loss))
            print('\r Validation Epoch_{}: DiceCoef={:.4f}'.format(epoch+1, te_coef))

            if best_dice < te_coef:
                best_dice = te_coef

        dice_list.append(best_dice)
        print('\r Fold_{}: DiceCoef={:.2f}'.format(f_id + 1, best_dice*100))

    print('\r Maximum performance: DiceCoef={:.2f}'.format(np.max(dice_list)*100))
    print('\r Average performance: DiceCoef={:.2f}+/-{:.2f}'.format(np.mean(dice_list)*100, np.std(dice_list)*100))

def main():
    Train_Eval()

if __name__ == "__main__":
    main()
    #nohup python3 -u trainer.py >> /data/tmpexec/tb_log/trainer.log 2>&1 &