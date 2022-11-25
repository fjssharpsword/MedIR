import torch
import numpy as np
from utils.losses.cox_loss import CoxLoss

class CoxPlusLoss(object):

    def __init__(self, alpha=10):
        self.alpha = alpha
        self.coxloss = CoxLoss()

    def __call__(self, results_dict, survival_time, state_label,interval_label,eps=1e-7):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        cox_loss = self.coxloss(results_dict, survival_time, state_label,interval_label)


        ci_loss = self.ciloss(results_dict, survival_time, state_label,interval_label)
        
        loss_cox_plus = cox_loss + self.alpha*ci_loss
        return loss_cox_plus


    def ciloss(self,results_dict, survival_time, state_label,interval_label):
        predict = results_dict['logits']
        predict = predict.view(-1)

        index = torch.argsort(survival_time, dim=0, descending=False)  # 在Batch Size 获取升序的索引
        # 匹配降序的索引
        # Xi
        risk = torch.gather(input=predict, dim=0, index=index)  # 根据OS的升序 改变predict 、oss 的顺序    患者 的生存时间 与风险值不对应
        label_state = torch.gather(input=state_label, dim=0, index=index)

        n = 0
        ci_loss = 0
        for i, l in enumerate(label_state):
            if l == 1 and i<len(label_state)-1:
                dr = risk[i] - risk
                ci_loss += torch.max(torch.tensor(0).to(risk.device),1-torch.exp(dr[i+1:])).sum()
                n += len(dr[i+1:])

        ci_loss = ci_loss / n

        return ci_loss