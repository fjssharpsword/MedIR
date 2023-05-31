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
from sklearn.metrics import confusion_matrix, f1_score
import pywt
from tensorboardX import SummaryWriter
#self-defined
from dsts.generator import build_dataset, dice_coef
from nets.sa_unet import build_unet, DiceLoss

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
        var_out = torch.where(var_out>0.5, 1, 0) #for dice loss
        pr_lbl = torch.cat((pr_lbl, var_out.cpu()), 0)

    te_loss = np.mean(te_loss)
    te_coef = dice_coef(gt_lbl, pr_lbl)

    #other metrics
    pr_lbl = torch.flatten(pr_lbl)
    gt_lbl = torch.flatten(gt_lbl)
    te_acc = (pr_lbl == gt_lbl).sum()/len(gt_lbl)
    tn, fp, fn, tp = confusion_matrix(gt_lbl.numpy(), pr_lbl.numpy()).ravel()
    te_sen = tp /(tp+fn)
    te_spe = tn /(tn+fp)
    te_f1 = 2*(te_sen*te_spe)/(te_sen+te_spe)

    return te_coef, te_acc, te_f1

def Train_Eval():

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    #log_writer = SummaryWriter('/data/tmpexec/tb_log')
    
    print('********************Train and validation********************')
    X, y = build_dataset(down_fq=250, seg_len=250) #time domain
    print('\r Sample number: {}'.format(len(y)))
    dataset = TensorDataset(torch.FloatTensor(X).unsqueeze(1), torch.LongTensor(y))
    kf_set = KFold(n_splits=10, shuffle=True).split(X, y)

    dice_list, acc_list, f1_list = [], [], []
    for f_id, (tr_idx, te_idx) in enumerate(kf_set):
        print('\r Fold {} train and validation.'.format(f_id + 1))
        #dataset
        te_sampler = SubsetRandomSampler(te_idx)
        te_dataloader = DataLoader(dataset, batch_size = 512, sampler=te_sampler)
        tr_sampler = SubsetRandomSampler(tr_idx)
        tr_dataloader = DataLoader(dataset, batch_size = 512, sampler=tr_sampler)
        
        #model 
        model = build_unet(in_ch=1, n_classes=1).to(device)  #conv-unet
        optimizer_model = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
        criterion = DiceLoss()   #nn.CrossEntropyLoss()
        #cross-validation
        best_dice, best_acc, best_f1 = 0.0, 0.0, 0.0
        for epoch in range(100):
            tr_loss = train_epoch(model, tr_dataloader, criterion, optimizer_model, device)
            lr_scheduler_model.step()  #about lr and gamma
            te_coef, te_acc, te_f1 = eval_epoch(model, te_dataloader, criterion, device)

            print('\r Train Epoch_{}: DiceLoss={:.4f}'.format(epoch+1, tr_loss))
            print('\r Validation Epoch_{}: DiceCoef={:.4f}, Accuracy={:.4f}, F1 Score={:.4f}'.format(epoch+1, te_coef, te_acc, te_f1))

            if best_dice < te_coef:
                best_dice = te_coef
                best_acc = te_acc
                best_f1 = te_f1
                if len(dice_list) == 0 or (len(dice_list) > 0 and best_dice > np.max(dice_list)):
                    torch.save(model.state_dict(), '/data/pycode/MedIR/EEG/SPSW/ckpts/ga_unet.pkl')
                    print(' Epoch: {} model has been already save!'.format(epoch+1))
       
        dice_list.append(best_dice)
        acc_list.append(best_acc)
        f1_list.append(best_f1)
        print('\r Fold_{}: DiceCoef={:.2f}, Accuracy={:.2f}, F1 Score={:.2f}'.format(f_id + 1, best_dice*100, best_acc*100, best_f1*100))

    print('\r Maximum performance: DiceCoef={:.2f}, Accuracy={:.2f}, F1 Score={:.2f}'.format(np.max(dice_list)*100, np.max(acc_list)*100, np.max(f1_list)*100))
    print('\r Average performance: DiceCoef={:.2f}+/-{:.2f}, \
                                   Accuracy={:.2f}+/-{:.2f}, \
                                   F1 Score={:.2f}+/-{:.2f}'.format(np.mean(dice_list)*100, np.std(dice_list)*100,\
                                                                    np.mean(acc_list)*100, np.std(acc_list)*100,\
                                                                    np.mean(f1_list)*100, np.std(f1_list)*100))

def main():
    Train_Eval()

if __name__ == "__main__":
    main()
    #nohup python3 -u trainer.py >> /data/tmpexec/tb_log/ga_unet.log 2>&1 &