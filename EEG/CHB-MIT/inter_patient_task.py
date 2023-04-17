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
from sklearn.model_selection import train_test_split
from ConvNet import EEGConvNet
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
#self-defined
from inter_datagenerator import get_intra_dataset

def train_epoch(model, dataloader, loss_fn, optimizer, device):

    tr_loss = []
    gt_lbl = torch.FloatTensor()
    pr_lbl = torch.FloatTensor()
    model.train()
    for eegs, lbls in dataloader:
        var_eeg = eegs.to(device)
        var_lbl = lbls.to(device)
        optimizer.zero_grad()
        var_out = model(var_eeg)
        loss = loss_fn(var_out,var_lbl)
        loss.backward()
        optimizer.step()
        tr_loss.append(loss.item())
        _, var_prd = torch.max(var_out.data, 1)
        gt_lbl = torch.cat((gt_lbl, lbls), 0)
        pr_lbl = torch.cat((pr_lbl, var_prd.cpu()), 0)

    tr_loss = np.mean(tr_loss)
    tn, fp, fn, tp = confusion_matrix(gt_lbl.numpy(), pr_lbl.numpy()).ravel()
    tr_sen = tp /(tp+fn)
    tr_spe = tn /(tn+fp)

    return tr_loss, tr_sen, tr_spe

def eval_epoch(model, dataloader, loss_fn, device):

    te_loss = []
    gt_lbl = torch.FloatTensor()
    pr_lbl = torch.FloatTensor()
    model.eval()
    for eegs, lbls in dataloader:
        var_eeg = eegs.to(device)
        var_lbl = lbls.to(device)
        var_out = model(var_eeg)
        loss = loss_fn(var_out,var_lbl)
        te_loss.append(loss.item())
        _, var_prd = torch.max(var_out.data, 1)
        gt_lbl = torch.cat((gt_lbl, lbls), 0)
        pr_lbl = torch.cat((pr_lbl, var_prd.cpu()), 0)

    te_loss = np.mean(te_loss)
    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    tn, fp, fn, tp = confusion_matrix(gt_lbl.numpy(), pr_lbl.numpy()).ravel()
    te_sen = tp /(tp+fn)
    te_spe = tn /(tn+fp)

    return te_loss, te_sen, te_spe

def Train_Eval():

    print('********************Build model********************')
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    
    model = EEGConvNet(in_ch = 18, num_classes=2).to(device)  
    optimizer_model = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
    criterion = nn.CrossEntropyLoss()
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    #log_writer = SummaryWriter('/data/tmpexec/tb_log')
   
    print('********************Train and validation********************')
    tr_dataloader = get_intra_dataset(batch_size=32, shuffle=True, num_workers=0, dst_type='train')
    te_dataloader = get_intra_dataset(batch_size=32, shuffle=False, num_workers=0, dst_type='test')
            
    best_sen, best_spe = 0.0, 0.0
    for epoch in range(20):
            
        tr_loss, tr_sen, tr_spe = train_epoch(model, tr_dataloader, criterion, optimizer_model, device)
        lr_scheduler_model.step()  #about lr and gamma
        te_loss, te_sen, te_spe = eval_epoch(model, te_dataloader, criterion, device)

        #log_writer.add_scalars('EEG/CHB-MIT/Loss', {'Train':tr_loss, 'Test':te_loss}, epoch+1)
        #log_writer.add_scalars('EEG/CHB-MIT/Sen', {'Train':tr_sen, 'Test':te_sen}, epoch+1)
        #log_writer.add_scalars('EEG/CHB-MIT/Spe', {'Train':tr_spe, 'Test':te_spe}, epoch+1)
        print('Train Epoch_{}: Sensitivity {:.4f}: Specificity: {:.4f}'.format(epoch+1, tr_sen, tr_spe))
        print('Val Epoch_{}: Sensitivity {:.4f}: Specificity: {:.4f}'.format(epoch+1, te_sen, te_spe))

        if te_sen > best_sen: 
            best_sen = te_sen
            best_spe = te_spe

    print('Sensitivity: {:.2f}'.format(best_sen*100))
    print('Specificity: {:.2f}'.format(best_spe*100))

def main():
    Train_Eval()

if __name__ == "__main__":
    main()
    #nohup python3 inter_patient_task.py > logs/intra_patient_task.log 2>&1 &