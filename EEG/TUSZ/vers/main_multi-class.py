import os
import numpy as np
import math
import pandas as pd
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
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import pywt
from torch.nn import functional as F
from tensorboardX import SummaryWriter
#self-defined
from nets.ConvNet import EEG1DConvNet
from nets.GCNNet import EEGDGCNN

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
    model.eval()
    gt_lbl = torch.FloatTensor()
    pr_prb = torch.FloatTensor()
    ev_loss = []
    for eegs, lbls in dataloader:
        var_eeg = eegs.to(device)
        var_lbl = lbls.to(device)
        var_out = model(var_eeg)
        loss = loss_fn(var_out,var_lbl)
        ev_loss.append(loss.item())
        gt_lbl = torch.cat((gt_lbl, lbls), 0)
        pr_prb = torch.cat((pr_prb, var_out.data.cpu()), 0)

    _, pr_lbl = torch.max(pr_prb, 1)
    ev_acc = accuracy_score(gt_lbl.numpy(), pr_lbl.numpy())
    ev_f1 = f1_score(gt_lbl.numpy(), pr_lbl.numpy(), average='weighted')
    ev_auc = roc_auc_score(gt_lbl.numpy(), pr_prb.numpy(), multi_class='ovo')

    return np.mean(ev_loss), ev_acc, ev_f1, ev_auc

def TUSZ_Test(PATH_TO_DST_ROOT, CKPT_PATH, sym_sz):
    #loading model
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    #model = EEGDGCNN(in_channels = 1, num_electrodes = 500, num_classes=len(sym_sz)).to(device)
    model = EEG1DConvNet(in_ch=22, num_classes=len(sym_sz)).to(device)
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained model: "+ CKPT_PATH)
    model.eval()#turn to evaluation mode
    #laoding dataset
    X, y = np.load(PATH_TO_DST_ROOT+'tusz_te_eeg.npy'), np.load(PATH_TO_DST_ROOT+'tusz_te_lbl.npy')
    print('\r Sample number: {}'.format(len(y)))
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    dataloader = DataLoader(dataset=dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    gt_lbl = torch.FloatTensor()
    pr_prb = torch.FloatTensor()
    for eegs, lbls in dataloader:
        var_eeg = eegs.to(device)
        var_out = model(var_eeg)
        gt_lbl = torch.cat((gt_lbl, lbls), 0)
        pr_prb = torch.cat((pr_prb, var_out.data.cpu()), 0)

    _, pr_lbl = torch.max(pr_prb, 1)
    te_acc = accuracy_score(gt_lbl.numpy(), pr_lbl.numpy())
    te_f1 = f1_score(gt_lbl.numpy(), pr_lbl.numpy(), average='weighted')
    te_auc = roc_auc_score(gt_lbl.numpy(), pr_prb.numpy(), multi_class='ovo')
    #output auc for each class
    #gt_np = F.one_hot(gt_lbl, num_classes=len(sym_sz)).numpy()
    #pred_np = pr_prb.numpy()
    #for i in range(len(sym_sz)):
    #    auc = roc_auc_score(gt_np[:, i], pred_np[:, i])
    #    print('\n AUC of {} = {:.2f}'.format(sym_sz[i], auc*100))

    return te_acc, te_f1, te_auc

def TUSZ_Train_Eval(PATH_TO_DST_ROOT, CKPT_PATH, sym_sz):

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    #log_writer = SummaryWriter('/data/tmpexec/tb_log')
   
    print('********************Train and validation********************')
    X, y = np.load(PATH_TO_DST_ROOT+'tusz_tr_eeg.npy'), np.load(PATH_TO_DST_ROOT+'tusz_tr_lbl.npy')
    print('\r Sample number: {}'.format(len(y)))
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    kf_set = KFold(n_splits=10,shuffle=True).split(X, y)

    acc_list, f1_list, auc_list = [], [], []
    for f_id, (tr_idx, te_idx) in enumerate(kf_set):
        print('\n Fold {} train and validation.'.format(f_id + 1))

        print('********************Build model********************')
        #model = EEGDGCNN(in_channels = 1, num_electrodes = 500, num_classes=len(sym_sz)).to(device)
        model = EEG1DConvNet(in_ch=22, num_classes=len(sym_sz)).to(device)  #CNN
        optimizer_model = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
        criterion = nn.CrossEntropyLoss()

        print('********************Load dataset********************')
        tr_sampler = SubsetRandomSampler(tr_idx)
        te_sampler = SubsetRandomSampler(te_idx)
        tr_dataloader = DataLoader(dataset, batch_size = 512, sampler=tr_sampler) #
        te_dataloader = DataLoader(dataset, batch_size = 512, sampler=te_sampler)
        
        best_acc = 0.0
        for epoch in range(10):
            
            tr_loss, tr_acc = train_epoch(model, tr_dataloader, criterion, optimizer_model, device)
            lr_scheduler_model.step()  #about lr and gamma
            _, ev_acc, ev_f1, ev_auc = eval_epoch(model, te_dataloader, criterion, device)

            print('\n Train Epoch_{}: Loss={:.4f}, Accuracy={:.4f}'.format(epoch+1, tr_loss, tr_acc))
            print('\n Validation Epoch_{}: Accuracy={:.4f}, F1_Score={:.4f}, ROC_AUC={:.4f}'.format(epoch+1, ev_acc, ev_f1, ev_auc))

            if ev_acc > best_acc:
                best_acc = ev_acc
                torch.save(model.state_dict(), CKPT_PATH)
                print(' Epoch: {} model has been already save!'.format(epoch+1))

        te_acc, te_f1, te_auc = TUSZ_Test(PATH_TO_DST_ROOT, CKPT_PATH, sym_sz)
        acc_list.append(te_acc)
        f1_list.append(te_f1)
        auc_list.append(te_auc)
        print('\n Testset in Fold_{}: Accuracy={:.2f}, F1_Score={:.2f}, ROC_AUC={:.2f}'.format(f_id + 1, te_acc*100, te_f1*100, te_auc*100))
        
    print('\n Average performance: Accuracy={:.2f}+/-{:.2f},\
                                   F1-score={:.2f}+/-{:.2f},\
                                   ROC_AUC={:.2f}+/-{:.2f}'.format(\
                                   np.mean(acc_list)*100, np.std(acc_list)*100,\
                                   np.mean(f1_list)*100, np.std(f1_list)*100,\
                                   np.mean(auc_list)*100, np.std(auc_list)*100))

def main():
    PATH_TO_DST_ROOT = '/data/pycode/MedIR/EEG/TUSZ/dsts/'
    sym_sz = ['bckg', 'cpsz', 'spsz', 'tnsz', 'mysz', 'tcsz', 'gnsz', 'fnsz']
    CKPT_PATH = '/data/pycode/MedIR/EEG/TUSZ/ckpts/convnet.pkl'
    TUSZ_Train_Eval(PATH_TO_DST_ROOT, CKPT_PATH, sym_sz)

if __name__ == "__main__":
    main()
    #nohup python3 -u main.py >> /data/tmpexec/tb_log/convnet.log 2>&1 &