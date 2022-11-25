
from utils.c_index import c_index,my_pycox_cindex,ours_cindex
import numpy as np
from pycox.evaluation import EvalSurv
import pandas as pd
from pycox.models.utils import pad_col
import torch
from sklearn.metrics import roc_curve, auc

    
def train(train_loader, optimizer, model,criterion,logger,cfg):
    model.train()
    w = torch.from_numpy(np.linspace(1.0,0.5,4)).float()
    total_loss = []
    tottal_cindex = []
    total_pred = None
    total_time = None
    total_label = None
    total_interval_label = None

    total_time_ipcw = None
    total_label_ipcw = None
    slide_ids = []
    for step, data in enumerate(train_loader):
        feature, survival_time, state_label, interval_label,slide_id = data
        slide_ids.extend(slide_id)
        if total_time_ipcw is None:
            total_time_ipcw = survival_time
            total_label_ipcw = state_label
        else:
            total_time_ipcw = torch.cat([survival_time, total_time_ipcw], dim=0)
            total_label_ipcw = torch.cat([state_label, total_label_ipcw], dim=0)


        if state_label.sum() < 2:
            continue

        optimizer.zero_grad()
        results_dict = model(data=feature.cuda())
        # loss = Cox_Loss(predict=results_dict['logits'], label_os=survival_label.cuda(),
                        # label_oss=state_label.cuda())

        loss = criterion(results_dict, survival_time.cuda(),state_label.cuda(),interval_label.cuda())

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

        

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

        if cfg.General.loss_name in ['CELoss','CE+Loss','OursLoss','NLLLoss']:
            # # deephit
            # pmf = pad_col(total_pred).softmax(1)[:,:-1]
            # surv = 1 - pmf.cumsum(1).detach().cpu().numpy()
            # eval_pred = pd.DataFrame(surv.transpose())
            # ev = EvalSurv(eval_pred, total_time.cpu().numpy(), total_label.cpu().numpy(), censor_surv='km')
            # pmf_cindex = ev.concordance_td('antolini')
            # my_cindex = my_pycox_cindex(surv,total_time,total_label)
            # assert pmf_cindex == my_cindex
            
            # risk = (pmf.detach().cpu() * w).sum(dim=1)
            # pmf = pmf.detach().cpu().numpy()
            pmf = pad_col(results_dict['logits']).softmax(1)[:,:-1]
            surv = 1 - pmf.cumsum(1).detach().cpu().numpy()
            score = ours_cindex(surv,survival_time,state_label,interval_label)
            tottal_cindex.append(score)
            # onehot = torch.zeros(pmf.shape)
            # onehot.scatter_(1, interval_label.long().view(-1,1), 1)
            # fpr,tpr,thresholds = roc_curve(onehot.ravel(),pmf.detach().cpu().ravel())
            # auc_score = auc(fpr,tpr)
            # print('auc {:.4f}'.format(auc_score))
            # _, category = pmf.topk(1,1,True,True)
            # category = category.view(-1)
            # print('num {} {} {} {}'.format(torch.where(category==0)[0].shape[0],torch.where(category==1)[0].shape[0],torch.where(category==2)[0].shape[0],torch.where(category==3)[0].shape[0]))
            # risk = torch.gather(total_pred.detach().cpu(), 1, total_interval_label.view(len(total_interval_label), 1).long())
            # risk = torch.gather(torch.from_numpy(surv), 1, total_interval_label.view(len(total_interval_label), 1).long())
            # risk = risk.view(-1)
            # pmf = pmf.detach().cpu().numpy()
            # total_data = np.stack([slide_ids,total_time.cpu().numpy(),total_label.cpu().numpy(),total_interval_label.cpu().numpy(),pmf[:,-4],pmf[:,-3],pmf[:,-2],pmf[:,-1]],axis=1).tolist()
            # df_feature = pd.DataFrame(total_data, columns=['slide_id','time','state','interval','interval','interval','interval','interval'])
            # df_feature.to_csv('train.csv',index=False)


        else:
            # pmf_cindex = 0
            risk = total_pred.sigmoid().squeeze()

            score = c_index(-risk, total_time, total_label)
            tottal_cindex.append(score)

        if step % 5 == 0:
            print('{}/{} loss={:.4f} cindex={:.4f}'.format(step+1,len(train_loader),loss.item(),score))
            logger.info('{}/{} loss={:.4f} cindex={:.4f}'.format(step+1,len(train_loader),loss.item(),score))

    # pmf = pad_col(total_pred).softmax(1)[:,:-1].detach().cpu().numpy()
    # total_data = np.stack([slide_ids,total_time.cpu().numpy(),total_label.cpu().numpy(),total_interval_label.cpu().numpy(),pmf[:,-4],pmf[:,-3],pmf[:,-2],pmf[:,-1]],axis=1).tolist()
    # df_feature = pd.DataFrame(total_data, columns=['slide_id','time','state','interval','interval','interval','interval','interval'])
    # df_feature.to_csv('train.csv',index=False)

    print('cindex={:.4f}'.format(np.mean(tottal_cindex)))
    logger.info(' cindex={:.4f}'.format(np.mean(tottal_cindex)))

    return np.mean(total_loss), np.mean(tottal_cindex),model,total_time_ipcw,total_label_ipcw