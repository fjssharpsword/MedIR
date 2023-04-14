import os
import numpy as np
from patient import Patient
from sklearn.model_selection import KFold
import torch
from torchvision import transforms
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from ConvNet import EEGConvNet
from sklearn.metrics import recall_score,roc_curve, auc
from tensorboardX import SummaryWriter

# Generates splits within a patient (defaults seconds =20s)
def intra_patient_split(patient_id, seconds = 20):

    p = Patient(patient_id)

    data = p.get_eeg_data()
    #labels = p.get_seizure_labels()
    win_len = int(p.get_sampling_rate() * seconds)

    #positive samples
    sei_data, sei_idx = [], []
    for sei_seg in p._seizure_intervals:
        num = int((sei_seg[1]-sei_seg[0])/win_len)
        pos = sei_seg[0]
        for i in range(num):
            pos = sei_seg[0]+i*win_len
            sei_data.append(data[pos:pos+win_len])
            sei_idx.append(1)
        if (sei_seg[1]-pos)>win_len/2:
            sei_data.append(data[pos:pos+win_len])
            sei_idx.append(1)

    #negative samples
    non_sei_data, non_sei_idx = [], []
    for i in range(len(p._seizure_intervals)-1):
        if len(non_sei_data) >= len(sei_data)*10: #pos:neg=1:10 
            break
        num = int((p._seizure_intervals[i+1][0]-p._seizure_intervals[i][1])/win_len)
        for j in range(num):
            pos = p._seizure_intervals[i][1]+j*win_len
            non_sei_data.append(data[pos:pos+win_len])
            non_sei_idx.append(0)
            if len(non_sei_data) >= len(sei_data)*10: #pos:neg=1:10 
                break

    #10-fold cross validataion
    X = np.array(sei_data + non_sei_data)
    y = np.array(sei_idx + non_sei_idx)

    return torch.FloatTensor(X).permute(0,2,1), torch.LongTensor(y), len(p.get_channel_names())

def train_epoch(model, tr_dataloader, loss_fn, optimizer, device):

    train_loss, train_correct=0.0, 0
    model.train()
    for eegs, lbls in tr_dataloader:
        var_eeg = eegs.to(device)
        var_lbl = lbls.to(device)
        optimizer.zero_grad()
        var_out = model(var_eeg)
        loss = loss_fn(var_out,var_lbl)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*var_eeg.size(0)
        _, predictions = torch.max(var_out.data, 1)
        train_correct += (predictions == var_lbl).sum().item()

    return train_loss, train_correct

def valid_epoch(model, te_dataloader, loss_fn, device):

    valid_loss, val_correct = 0.0, 0
    model.eval()

    te_lbl = torch.FloatTensor()
    pr_lbl = torch.FloatTensor()

    with torch.no_grad():

        for eegs, lbls in te_dataloader:
            var_eeg = eegs.to(device)
            var_lbl = lbls.to(device)
            var_out = model(var_eeg)
            loss = loss_fn(var_out,var_lbl)
            valid_loss+=loss.item()*var_eeg.size(0)
            _, predictions = torch.max(var_out.data,1)
            val_correct+=(predictions == var_lbl).sum().item() #accuracy
            te_lbl = torch.cat((te_lbl, lbls), 0)
            pr_lbl = torch.cat((pr_lbl, predictions.cpu()), 0)

    val_recall = recall_score(te_lbl.numpy(), pr_lbl.numpy())* 100 #recall

    return valid_loss, val_correct, val_recall 

def Train_Test():

    log_writer = SummaryWriter('/data/tmpexec/tb_log')
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    for  p_id in [id for id in range(1, 2)]: # 24 cases
        print('Patient {} train and validation.'.format(p_id))

        X, y, in_ch = intra_patient_split(patient_id=p_id, seconds = 20)

        X_train, X_test = train_test_split(X)

        print('********************Build model********************')
        device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
        model = EEGConvNet(in_ch = in_ch, num_classes=2).to(device)  
        optimizer_model = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
        lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
        criterion = nn.CrossEntropyLoss()

        print('********************Train and validation********************')
        dataset = TensorDataset(X, y)
        kf_set = KFold(n_splits=10,shuffle=True).split(X, y)
        te_acc, te_rec = [], []
        for f_id, (tr_idx, te_idx) in enumerate(kf_set):
            print('Fold {} train and validation.'.format(f_id + 1))
            tr_sampler = SubsetRandomSampler(tr_idx)
            te_sampler = SubsetRandomSampler(te_idx)
            tr_dataloader = DataLoader(dataset, batch_size = 32, sampler=tr_sampler) 
            te_dataloader = DataLoader(dataset, batch_size = 32, sampler=te_sampler) 
            
            best_te_acc, best_te_recall = 0.0, 0.0
            for epoch in range(20):
                tr_loss, tr_correct=train_epoch(model, tr_dataloader, criterion, optimizer_model, device)
                lr_scheduler_model.step()  #about lr and gamma
                te_loss, te_correct, te_recall = valid_epoch(model,te_dataloader,criterion, device)

                tr_loss = tr_loss / len(tr_dataloader.sampler)
                tr_correct = tr_correct / len(tr_dataloader.sampler) * 100
                te_loss = te_loss / len(te_dataloader.sampler)
                te_correct = te_correct / len(te_dataloader.sampler) * 100
                
                log_writer.add_scalars('Patient_{}/Fold_{}/Loss'.format(p_id, f_id), {'Train':tr_loss, 'Test':te_loss}, epoch+1)
                log_writer.add_scalars('Patient_{}/Fold_{}/Acc'.format(p_id, f_id), {'Train':tr_correct, 'Test':te_correct}, epoch+1)

                if te_correct > best_te_acc: best_te_acc = te_correct
                if te_recall > best_te_acc: best_te_recall = te_recall
            te_acc.append(best_te_acc)
            te_rec.append(best_te_recall)
            print('Accuracy of fold {} fold for patient {} cross validation: {:.2f}'.format(f_id, p_id, best_te_acc))
            print('Recall of fold {} fold for patient {} cross validation: {:.2f}'.format(f_id, p_id, best_te_recall))
        print('Accuracy of patient {}: {:.2f} +/- {:.2f}'.format(p_id, np.mean(te_acc), np.std(te_acc)))
        print('Recall of patient {}: {:.2f} +/- {:.2f}'.format(p_id, np.mean(te_rec), np.std(te_rec)))

def main():
    Train_Test()

if __name__ == "__main__":
    main()
    #nohup python3 intra_patient_task.py > logs/intra_patient_task.log 2>&1 &