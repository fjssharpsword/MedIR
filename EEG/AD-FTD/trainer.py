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
from sklearn.metrics import confusion_matrix, roc_auc_score
from tensorboardX import SummaryWriter
#self-defined
from datagenerator import get_dataset
from ConvNet import EEG1DConvNet
from LSTMNet import EEGLSTM

def train_epoch(model, dataloader, loss_fn, optimizer, device):

    tr_loss = []
    tr_acc = 0.0
    gt_lbl = torch.FloatTensor()
    model.train()
    for eegs, lbls in dataloader:

        #eegs = eegs[0].unsqueeze(0)
        #lbls = lbls[0].unsqueeze(0)

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
        tr_acc += (var_prd == var_lbl).sum().item()

    tr_loss = np.mean(tr_loss)
    tr_acc = tr_acc/len(gt_lbl)

    return tr_loss, tr_acc

def eval_epoch(model, dataloader, loss_fn, device):

    te_loss = []
    gt_lbl = torch.FloatTensor()
    pr_lbl = torch.FloatTensor()
    te_acc = 0.0
    model.eval()
    for eegs, lbls in dataloader:

        #eegs = eegs[0].unsqueeze(0)
        #lbls = lbls[0].unsqueeze(0)
        
        var_eeg = eegs.to(device)
        var_lbl = lbls.to(device)
        var_out = model(var_eeg)
        loss = loss_fn(var_out,var_lbl)
        te_loss.append(loss.item())
        _, var_prd = torch.max(var_out.data, 1)
        gt_lbl = torch.cat((gt_lbl, lbls), 0)
        pr_lbl = torch.cat((pr_lbl, var_prd.cpu()), 0)
        te_acc += (var_prd == var_lbl).sum().item()

    te_loss = np.mean(te_loss)
    #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

    results = confusion_matrix(gt_lbl.numpy(), pr_lbl.numpy()).ravel()
    if len(results) == 4:
        tn, fp, fn, tp = results
        te_sen = tp /(tp+fn)
        te_spe = tn /(tn+fp)
    else:
        te_sen = 0.50
        te_spe = 0.50

    te_acc = te_acc/len(gt_lbl)

    return te_loss, te_sen, te_spe, te_acc

def Train_Eval():

    print('********************Build model********************')
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    
    #model = EEG1DConvNet(in_ch = 19, num_classes=3).to(device) 
    model = EEGLSTM(num_electrodes = 19, hid_channels=64, num_classes=3).to(device) 
    
    optimizer_model = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
    criterion = nn.CrossEntropyLoss()
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
   
    print('********************Train and validation********************')
    te_dataloader = get_dataset(batch_size=32, shuffle=False, num_workers=0, dst_type='te')
    best_sen, best_spe = 0.0, 0.0
    best_acc = 0.0 
    for epoch in range(30):
        tr_dataloader = get_dataset(batch_size=32, shuffle=True, num_workers=0, dst_type='tr')
    
        tr_loss, tr_acc = train_epoch(model, tr_dataloader, criterion, optimizer_model, device)
        lr_scheduler_model.step()  #about lr and gamma
        _, te_sen, te_spe, te_acc = eval_epoch(model, te_dataloader, criterion, device)

        print('\n Train Epoch_{}: Loss={:.4f}, Accuracy={:.4f}'.format(epoch+1, tr_loss, tr_acc))
        print('\n Validation Epoch_{}: Accuracy={:.4f}, Sensitivity={:.4f}, Specificity={:.4f}'.format(epoch+1, te_acc, te_sen, te_spe))

        if te_sen > best_sen:
            best_acc = te_acc
            best_sen = te_sen
            best_spe = te_spe

    print('\n Accuracy={:.2f}, Sensitivity={:.2f}, Specificity={:.2f}'.format(best_acc*100, best_sen*100, best_spe*100))

def main():
    Train_Eval()

if __name__ == "__main__":
    main()
    #nohup python3 trainer.py > logs/intra_patient_task.log 2>&1 &