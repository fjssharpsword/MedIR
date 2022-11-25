
from utils.c_index import c_index,ours_cindex
import torch,os
import numpy as np
from pycox.models.utils import pad_col
from pycox.evaluation import EvalSurv
import pandas as pd
from sksurv.metrics import cumulative_dynamic_auc
import sksurv
from sklearn.metrics import cohen_kappa_score,accuracy_score
from sklearn.metrics import roc_curve, auc


def transform_to_struct_array(times, events):
    return sksurv.util.Surv.from_arrays(events, times)

def get_optimal_cutoff(TPR, FPR, threshold):
    print(len(TPR),len(threshold))
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def get_auc(train,test,risk,times=60):
    durations_train, events_train = train
    durations_test, events_test = test
    durations_train = torch.cat([durations_train,durations_test],0)
    events_train = torch.cat([events_train,events_test],0)
    risk = risk.detach().cpu().numpy()
    # print(len(durations_train),max(durations_train),sum(events_train))
    auc,mean_auc,threshold,tpr,fpr = cumulative_dynamic_auc(transform_to_struct_array(durations_train, events_train), transform_to_struct_array(durations_test, events_test),risk,times)
    
    return auc,mean_auc,threshold,tpr,fpr

def get_kappa_and_acc(survival_time,survival_state,risk,optimal_threshold,times=60):
    risk = risk.detach().cpu().numpy()
    tmp_y_label = []
    for t,s in zip(survival_time,survival_state):
        if s==1 and t <= times:
            label = 1
        elif s==0 and t <= times:
            label = -1
        else: # t > cut_of_time:
            label = 0
        
        tmp_y_label.append(label)

    y_pred = []
    y_label = []
    for i, label in enumerate(tmp_y_label):
        if label != -1:
            y_pred.append(1 if risk[i] >= optimal_threshold else 0)
            y_label.append(label)

    kappa = cohen_kappa_score(y_label,y_pred)
    acc = accuracy_score(y_label,y_pred)

    return kappa,acc


def val_test(val_loader, model,criterion,cfg,train_total_time,train_total_label):
    model.eval()
    w = torch.from_numpy(np.linspace(1.0,0.5,4)).float()
    with torch.no_grad():
        # total_loss = []
        total_pred = None
        total_time = None
        total_label = None
        total_interval_label = None
        slide_ids = []
        for step, data in enumerate(val_loader):
            feature, survival_time, state_label, interval_label,slide_id = data
            slide_ids.extend(slide_id)
            # if 'TCGA-52-7812-01Z-00-DX1.dd6fa49a-f9fe-40a1-80b2-0824b128f3b2' in slide_id:
            #     print(slide_id)
            results_dict = model(data=feature.cuda())

            if total_pred is None:
                total_pred = results_dict['logits']
                total_time = survival_time
                total_label = state_label
                total_interval_label = interval_label
            else:
                total_pred = torch.cat([results_dict['logits'], total_pred], dim=0)
                total_time = torch.cat([survival_time, total_time], dim=0)
                total_label = torch.cat([state_label, total_label], dim=0)
                total_interval_label = torch.cat([interval_label, total_interval_label], dim=0)

            # loss = Cox_Loss(predict=results_dict['logits'], label_os=survival_time.cuda(),
            #                 label_oss=state_label.cuda())
        loss = criterion({'logits':total_pred}, total_time.cuda(),total_label.cuda(),total_interval_label.cuda())


        total_loss = loss.item()

        if cfg.General.loss_name in ['CELoss','CE+Loss','OursLoss','NLLLoss']:
            # deephit
            # pmf = pad_col(total_pred).softmax(1)[:,:-1]
            # surv = 1 - pmf.cumsum(1).detach().cpu().numpy()
            # eval_pred = pd.DataFrame(surv.transpose())
            # ev = EvalSurv(eval_pred, total_time.long().cpu().numpy(), total_label.cpu().numpy(), censor_surv='km')
            # pmf_cindex = ev.concordance_td('antolini')

            # risk = (pmf.detach().cpu() * w).sum(dim=1)
            # pmf = pmf.detach().cpu().numpy()
            # total_pred = pad_col(total_pred).softmax(1)[:,:-1]
            # # risk, category = total_pred.topk(1,1,True,True)
            # risk = torch.gather(total_pred.detach().cpu(), 1, total_interval_label.view(len(total_interval_label), 1).long())
            # risk = risk.view(-1)
            total_pred = pad_col(total_pred).softmax(1)[:,:-1]
            surv = 1 - total_pred.cumsum(1).detach().cpu().numpy()
            score = ours_cindex(surv,total_time,total_label,total_interval_label)

            _, category = total_pred.topk(1,1,True,True)
            category = category.view(-1)
            print('num {} {} {} {}'.format(torch.where(category==0)[0].shape[0],torch.where(category==1)[0].shape[0],torch.where(category==2)[0].shape[0],torch.where(category==3)[0].shape[0]))
        

            pmf = total_pred.detach().cpu().numpy()
            
            total_data = np.stack([slide_ids,total_time.cpu().numpy(),total_label.cpu().numpy(),total_interval_label.cpu().numpy(),pmf[:,-4],pmf[:,-3],pmf[:,-2],pmf[:,-1]],axis=1).tolist()

       
        else:
            
            risk = total_pred.sigmoid().squeeze()

            total_data = np.stack([slide_ids,total_time.cpu().numpy(),total_label.detach().cpu().numpy(),risk.detach().cpu().numpy(),risk.detach().cpu().numpy(),risk.detach().cpu().numpy(),risk.detach().cpu().numpy(),risk.detach().cpu().numpy()],axis=1).tolist()
    

        
            score = c_index(-risk, total_time, total_label)

        # auc,mean_auc,threshold,tpr,fpr = get_auc([train_total_time,train_total_label],[total_time,total_label],risk,times=60)
        # optimal_threshold, point = get_optimal_cutoff(tpr, fpr, threshold)
        # kappa,acc = get_kappa_and_acc(total_time,total_label,risk,optimal_threshold,times=60)
        mean_auc = 0
        kappa = 0
        tpr = [0]
        fpr = [0]
        optimal_threshold = 0
        acc = 0


           
        

    return total_loss, score,total_data,mean_auc,kappa,tpr,fpr,optimal_threshold,acc