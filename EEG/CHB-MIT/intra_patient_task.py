import os
import numpy as np
import math
from patient import Patient
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

# Generates splits within a patient (defaults seconds =20s)
def intra_patient_split(patient_id, seconds = 20):

    p = Patient(patient_id)
    data = p.get_eeg_data()
    #labels = p.get_seizure_labels()
    win_len = int(p.get_sampling_rate() * seconds)

    #positive samples
    sei_data= []
    for sei_seg in p._seizure_intervals:
        num = int((sei_seg[1]-sei_seg[0])/win_len)
        pos = sei_seg[0]
        for i in range(num):
            pos = sei_seg[0]+i*win_len
            sei_data.append(data[pos:pos+win_len])
        if (sei_seg[1]-pos)>win_len/2:
            sei_data.append(data[pos:pos+win_len])
    random.shuffle(sei_data)
    n_sei_te = max(1, int(math.ceil(0.2 * len(sei_data))))
    n_sei_tr = len(sei_data) - n_sei_te
    X_sei_tr = sei_data[:n_sei_tr]
    y_sei_tr = np.ones(n_sei_tr)
    X_sei_te = sei_data[n_sei_tr:]
    y_sei_te = np.ones(n_sei_te)

    #negative samples
    non_sei_data = []
    for i in range(len(p._seizure_intervals)-1):
        if len(non_sei_data) >= len(sei_data)*10: #pos:neg=1:10 
            break
        num = int((p._seizure_intervals[i+1][0]-p._seizure_intervals[i][1])/win_len)
        for j in range(num):
            pos = p._seizure_intervals[i][1]+j*win_len
            non_sei_data.append(data[pos:pos+win_len])
            if len(non_sei_data) >= len(sei_data)*10: #pos:neg=1:10 
                break
    random.shuffle(non_sei_data)
    n_non_sei_te = max(1, int(round(0.2 * len(non_sei_data))))
    n_non_sei_tr = len(non_sei_data) - n_non_sei_te
    X_non_sei_tr = non_sei_data[:n_non_sei_tr]
    y_non_sei_tr = np.zeros(n_non_sei_tr)
    X_non_sei_te = non_sei_data[n_non_sei_tr:]
    y_non_sei_te = np.zeros(n_non_sei_te)

    #split trainset and testset
    X_tr, y_tr = np.array(X_sei_tr + X_non_sei_tr), np.append(y_sei_tr, y_non_sei_tr)
    tr_dataset = TensorDataset(torch.FloatTensor(X_tr).permute(0,2,1), torch.LongTensor(y_tr))
    X_te, y_te = np.array(X_sei_te + X_non_sei_te), np.append(y_sei_te, y_non_sei_te)
    te_dataset = TensorDataset(torch.FloatTensor(X_te).permute(0,2,1), torch.LongTensor(y_te))

    return tr_dataset, te_dataset, len(p.get_channel_names()[0])

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

    return tr_loss, tr_sen*100, tr_spe*100

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

    log_writer = SummaryWriter('/data/tmpexec/tb_log')
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    for  p_id in [id for id in range(24, 25)]: # range(1,25)-24 cases
        print('Patient_{} train and validation.'.format(p_id))
        tr_dataset, te_dataset, in_ch = intra_patient_split(patient_id=p_id, seconds = 20)

        print('********************Build model********************')
        device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
        model = EEGConvNet(in_ch = in_ch, num_classes=2).to(device)  
        optimizer_model = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
        criterion = nn.CrossEntropyLoss()

        print('********************Train and validation********************')
        te_dataloader = DataLoader(te_dataset, batch_size = 32, shuffle=True) 
        sens, spes = [], []
        for f_id in range(10):
            print('Patient_{}: Fold_{}: Start Training.'.format(p_id, f_id + 1))
            tr_dataloader = DataLoader(tr_dataset, batch_size = 32, shuffle=True)
            
            best_sen, best_spe = 0.0, 0.0
            for epoch in range(20):
                tr_loss, tr_sen, tr_spe = train_epoch(model, tr_dataloader, criterion, optimizer_model, device)
                lr_scheduler_model.step()  #about lr and gamma
                te_loss, te_sen, te_spe = eval_epoch(model, te_dataloader, criterion, device)

                log_writer.add_scalars('Patient_{}/Fold_{}/Loss'.format(p_id, f_id + 1), {'Train':tr_loss, 'Test':te_loss}, epoch+1)
                log_writer.add_scalars('Patient_{}/Fold_{}/Sen'.format(p_id, f_id + 1), {'Train':tr_sen, 'Test':te_sen}, epoch+1)
                log_writer.add_scalars('Patient_{}/Fold_{}/Spe'.format(p_id, f_id + 1), {'Train':tr_spe, 'Test':te_spe}, epoch+1)

                if te_sen > best_sen: best_sen = te_sen
                if te_spe > best_spe: best_spe = te_spe

            print('Patient_{}: Fold_{}: Sensitivity: {:.6f}'.format(p_id, f_id + 1, best_sen))
            print('Patient_{}: Fold_{}: Specificity: {:.6f}'.format(p_id, f_id + 1, best_spe))
            sens.append(best_sen)
            spes.append(best_spe)

        print('Patient_{}: Sensitivity: {:.2f} +/- {:.2f}'.format(p_id, np.mean(sens)*100, np.std(sens)*100))
        print('Patient_{}: Specificity: {:.2f} +/- {:.2f}'.format(p_id, np.mean(spes)*100, np.std(spes)*100))

def main():
    Train_Eval()

if __name__ == "__main__":
    main()
    #nohup python3 intra_patient_task.py > logs/intra_patient_task.log 2>&1 &