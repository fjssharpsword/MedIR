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
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from tensorboardX import SummaryWriter
#self-defined
from dsts.tuev_spsw import build_dataset
from nets.ConvNet import EEG1DConvNet

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

    pr_prb, pr_lbl = torch.max(pr_prb, 1)
    ev_acc = accuracy_score(gt_lbl.numpy(), pr_lbl.numpy())
    ev_f1 = f1_score(gt_lbl.numpy(), pr_lbl.numpy(), average='weighted')
    ev_auc = roc_auc_score(gt_lbl.numpy(), pr_prb.numpy())

    return np.mean(ev_loss), ev_acc, ev_f1, ev_auc

def Train_Eval(CKPT_PATH):

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    #log_writer = SummaryWriter('/data/tmpexec/tb_log')
    
    print('********************Train and validation********************')
    X, y = build_dataset(down_fq=250, seg_len=250) #time domain
    print('\r Sample number: {}'.format(len(y)))
    dataset = TensorDataset(torch.FloatTensor(X).unsqueeze(1), torch.LongTensor(y))
    kf_set = KFold(n_splits=10, shuffle=True).split(X, y)

    acc_list, f1_list, auc_list = [], [], []
    for f_id, (tr_idx, te_idx) in enumerate(kf_set):
        print('\r Fold {} train and validation.'.format(f_id + 1))
        #dataset
        te_sampler = SubsetRandomSampler(te_idx)
        te_dataloader = DataLoader(dataset, batch_size = 512, sampler=te_sampler)
        tr_sampler = SubsetRandomSampler(tr_idx)
        tr_dataloader = DataLoader(dataset, batch_size = 512, sampler=tr_sampler)
        
        #model 
        model = EEG1DConvNet(in_ch=1, num_classes=2).to(device)  #CNN
        optimizer_model = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
        criterion = nn.CrossEntropyLoss()

        #cross-validation
        best_acc, best_f1, best_auc = 0.0, 0.0, 0.0
        for epoch in range(10):
            
            tr_loss, tr_acc = train_epoch(model, tr_dataloader, criterion, optimizer_model, device)
            lr_scheduler_model.step()  #about lr and gamma
            _, ev_acc, ev_f1, ev_auc = eval_epoch(model, te_dataloader, criterion, device)

            print('\n Train Epoch_{}: Loss={:.4f}, Accuracy={:.4f}'.format(epoch+1, tr_loss, tr_acc))
            print('\n Validation Epoch_{}: Accuracy={:.4f}, F1_Score={:.4f}, ROC_AUC={:.4f}'.format(epoch+1, ev_acc, ev_f1, ev_auc))

            if ev_acc > best_acc:
                best_acc = ev_acc
                best_f1 = ev_f1
                best_auc = ev_auc
                torch.save(model.state_dict(), CKPT_PATH)
                print(' Epoch: {} model has been already save!'.format(epoch+1))

        acc_list.append(best_acc)
        f1_list.append(best_f1)
        auc_list.append(best_auc)
        print('\n Testset in Fold_{}: Accuracy={:.2f}, F1_Score={:.2f}, ROC_AUC={:.2f}'.format(f_id + 1, best_acc*100, best_f1*100, best_auc*100))
        
    print('\n Average performance: Accuracy={:.2f}+/-{:.2f},\
                                   F1-score={:.2f}+/-{:.2f},\
                                   ROC_AUC={:.2f}+/-{:.2f}'.format(\
                                   np.mean(acc_list)*100, np.std(acc_list)*100,\
                                   np.mean(f1_list)*100, np.std(f1_list)*100,\
                                   np.mean(auc_list)*100, np.std(auc_list)*100))
def main():
    CKPT_PATH = '/data/pycode/MedIR/EEG/TUSZ/ckpts/convnet.pkl'
    Train_Eval(CKPT_PATH)

if __name__ == "__main__":
    main()
    #nohup python3 -u tuev_trainer.py >> /data/tmpexec/tb_log/tuev_trainer.log 2>&1 &