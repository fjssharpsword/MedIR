import torch
import numpy as np
from pycox.models.utils import pad_col

class CoxLoss(object):

    def __init__(self, alpha=0.15):
        self.alpha = alpha
        # self.w = torch.from_numpy(np.linspace(1.0,0.5,4)).float()

    # def __call__(self, results_dict, survival_time, state_label,interval_label,eps=1e-7):
    #     # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    #     # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    #     hazards = results_dict['logits']#.sigmoid()
    #     # hazards = pad_col(hazards,val=1)
    #     # S = torch.cumprod(1 - hazards, dim=1)#[:,-1]
    #     # hazards = 1-torch.sum(S,dim=1)
    #     # hazards = (hazards * self.w.to(hazards.device)).sum(dim=1)

    #     batch_size = len(survival_time)
       
    #     # sta_label = state_label.view(batch_size, 1).float()
        


    #     # current_batch_len = len(Survs)
    #     R_mat = np.zeros([batch_size, batch_size], dtype=int)
    #     for i in range(batch_size):
    #         for j in range(batch_size):
    #             R_mat[i,j] = survival_time[j] > survival_time[i]

    #     R_mat = torch.FloatTensor(R_mat).cuda()
    #     theta = hazards.reshape(-1)
    #     exp_theta = torch.exp(theta)
    #     sum_exp = torch.sum(exp_theta*R_mat, dim=1)
    #     index = torch.BoolTensor(sum_exp.detach().cpu().numpy())
    #     loss_cox = -torch.sum(((theta - torch.log(sum_exp)) * (state_label))[index])/torch.sum(state_label[index])
    #     return loss_cox

    
    def __call__(self, results_dict, survival_time, state_label,interval_label,eps=1e-7):
        predict = results_dict['logits']
        predict = predict.view(-1)

        index = torch.argsort(survival_time, dim=0, descending=True)  # 在Batch Size 获取降序的索引
        # 匹配降序的索引
        # Xi
        risk = torch.gather(input=predict, dim=0, index=index)  # 根据OS的降序 改变predict 、oss 的顺序    患者 的生存时间 与风险值不对应
        label_state = torch.gather(input=state_label, dim=0, index=index)
        # label_os = torch.sort(label_os,dim=0,descending=True).values
        # torch.exp(x) y=e^x
        # 风险
        hazard_ratio = torch.exp(risk)
        '''
        a=[1,2,3]
        torch.cumsum:累加,b=[1,3,6]
        '''
        # XI,log e^x = x
        # log i:ti≥tj的样本的风险累加和
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))  # 包括本身的风险值
        # temp = torch.ones_like(hazard_ratio).cuda()
        # temp = torch.cumsum(temp, dim=0)
        # log_risk = torch.log(torch.cumsum(hazard_ratio / temp, dim=0)) # 包括本身的风险值

        # Rj-log i:ti≥tj的样本的风险累加和 参考硕士论文公式(4-2)
        partial_likelihood = risk - log_risk
        # 观察到复发的样本label_state=1
        uncensored_likelihood = partial_likelihood * label_state  # 除去无结局的样本

        num_observed_events = torch.sum(label_state)
        # 如果mini-batch 都为无结局

        num_observed_events = num_observed_events.float()
        # 合并a,b两个tensor a>0的地方保存，防止分母为0
        # 如果全为负样本，0->1e-7
        num_observed_events = torch.where(num_observed_events > 0, num_observed_events,
                                        torch.tensor(1e-7, device=num_observed_events.device, ))
        # 类似除以batch size
        loss_cox = -torch.sum(uncensored_likelihood) / num_observed_events

        return loss_cox
