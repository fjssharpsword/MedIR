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
from sklearn.metrics import confusion_matrix
import pywt
from tensorboardX import SummaryWriter
#self-defined
from ConvNet import EEG2DConvNet, EEG1DConvNet
from LSTMNet import EEGLSTM
from GCNNet import EEGDGCNN
from nets.tsception import TSCeption
from nets.vanilla_transformer import VanillaTransformer
from nets.arjun_vit import ArjunViT

PATH_TO_DST_ROOT = '/data/pycode/MedIR/EEG/CHB-MIT/dsts/'
def train_epoch(model, dataloader, loss_fn, optimizer, device):

    tr_loss = []
    tr_acc = 0.0
    gt_lbl = torch.FloatTensor()
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
    tn, fp, fn, tp = confusion_matrix(gt_lbl.numpy(), pr_lbl.numpy()).ravel()
    te_sen = tp /(tp+fn)
    te_spe = tn /(tn+fp)

    te_acc = te_acc/len(gt_lbl)

    return te_loss, te_sen, te_spe, te_acc

def Train_Eval():

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    #log_writer = SummaryWriter('/data/tmpexec/tb_log')
    
    print('********************Train and validation********************')
    X, y = np.load(PATH_TO_DST_ROOT+'eeg_kfold_2s.npy'), np.load(PATH_TO_DST_ROOT+'lbl_kfold_2s.npy') #time domain

    #X = np.fft.fft(X, axis=1) #Fourier transform, frequence domian

    X_cA, X_cD = pywt.dwt(X, 'haar', mode='symmetric', axis=1) #wavelet transform, time-frequence domain
    X = np.concatenate((X_cA, X_cD), axis=1)

    dataset = TensorDataset(torch.FloatTensor(X).permute(0,2,1), torch.LongTensor(y))
    kf_set = KFold(n_splits=10,shuffle=True).split(X, y)
    sen_list, spe_list, acc_list = [], [], []
    for f_id, (tr_idx, te_idx) in enumerate(kf_set):
        print('\n Fold {} train and validation.'.format(f_id + 1))
        
        #model = EEG1DConvNet(in_ch = 18, num_classes=2).to(device)  #CNN
        #model = EEGLSTM(num_electrodes = 18, hid_channels=64, num_classes=2).to(device) #RNN
        #model = EEGDGCNN(in_channels = 18, num_electrodes = 512, num_classes=2).to(device) #GCN

        #model = TSCeption(num_electrodes=18, num_classes=2).to(device)
        #model = VanillaTransformer(num_electrodes=18, chunk_size=512, num_classes=2).to(device)
        model = ArjunViT(num_electrodes=18, chunk_size=512, num_classes=2).to(device)

        optimizer_model = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
        criterion = nn.CrossEntropyLoss()
        
        best_sen, best_spe, best_acc = 0.0, 0.0, 0.0
        for epoch in range(100):
            tr_sampler = SubsetRandomSampler(tr_idx)
            te_sampler = SubsetRandomSampler(te_idx)
            tr_dataloader = DataLoader(dataset, batch_size = 128, sampler=tr_sampler) #20s-128, 0.5s-512
            te_dataloader = DataLoader(dataset, batch_size = 128, sampler=te_sampler)
                
            tr_loss, tr_acc = train_epoch(model, tr_dataloader, criterion, optimizer_model, device)
            lr_scheduler_model.step()  #about lr and gamma
            _, te_sen, te_spe, te_acc = eval_epoch(model, te_dataloader, criterion, device)

            #log_writer.add_scalars('EEG/CHB-MIT/Loss', {'Train':tr_loss, 'Test':te_loss}, epoch+1)
            print('\n Train Epoch_{}: Loss={:.4f}, Accuracy={:.4f}'.format(epoch+1, tr_loss, tr_acc))
            print('\n Validation Epoch_{}: Accuracy={:.4f}, Sensitivity={:.4f}, Specificity={:.4f}'.format(epoch+1, te_acc, te_sen, te_spe))

            if te_sen > best_sen:
                best_acc = te_acc
                best_sen = te_sen
                best_spe = te_spe

        sen_list.append(best_sen)
        spe_list.append(best_spe)
        acc_list.append(best_acc)

        print('\n Fold_{}: Accuracy={:.2f}, Sensitivity={:.2f}, Specificity={:.2f}'.format(f_id + 1, best_acc*100, best_sen*100, best_spe*100))

    idx = np.argmax(acc_list)
    print('\n Maximum performance: Accuracy={:.2f}, Sensitivity={:.2f}, Specificity={:.2f}'.format(np.max(acc_list)*100,  sen_list[idx]*100, spe_list[idx]*100))
    print('\n Average performance: Accuracy={:.2f}+/-{:.2f},\
                                   Sensitivity={:.2f}+/-{:.2f},\
                                   Specificity={:.2f}+/-{:.2f}'.format(\
                                   np.mean(acc_list)*100, np.std(acc_list)*100,\
                                   np.mean(sen_list)*100, np.std(sen_list)*100,\
                                   np.mean(spe_list)*100, np.std(spe_list)*100))

def main():
    Train_Eval()

if __name__ == "__main__":
    main()
    #nohup python3 inter_patient_task_TF.py > logs/intra_patient_task.log 2>&1 &