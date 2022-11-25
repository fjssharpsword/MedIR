import torch
import numpy as np
from pycox.models.utils import pad_col

class TMCLoss(object):

    def __init__(self, beta=1,alpha_ij=0.5,alpha_i=0.25,alpha_j=0.25):
        self.alpha_ij = alpha_ij
        self.alpha_i = alpha_i
        self.alpha_j = alpha_j
        self.beta = beta

    def __call__(self, results_dict, survival_time, state_label,interval_label,eps=1e-7):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        hazards = results_dict['logits']
        hazards = pad_col(hazards).softmax(1)
        # .softmax(1)
        batch_size = len(survival_time)

        index = torch.argsort(survival_time, dim=0, descending=False)  # 在Batch Size 获取升序的索引
        label_time = torch.gather(input=survival_time, dim=0, index=index)
        # 匹配降序的索引
        # Xi
        risk = hazards[index]  # 根据OS的升序 改变predict 、oss 的顺序    患者 的生存时间 与风险值不对应
        label_state = torch.gather(input=state_label, dim=0, index=index)

        tmc_loss = []
        for i, l in enumerate(label_state):
            if l == 1 and i<len(label_state)-1:
                dis_t = (label_time - label_time[i])

                # for i
                indices = interval_label[i].long()
                f_i = risk[i,indices]
                f_i_sum = torch.sum(risk[i,:indices+1],dim=0)

                #for j
                f_y_1_sum = []
                f_y_1 = []
                f_y_0_sum = []
                f_y_0_0_sum = []
                for b in range(batch_size):
                    indices_j = interval_label[b].long()
                    f_y_1.append(risk[b,indices_j])
                    sample = torch.sum(risk[b,:indices_j+1],dim=0)
                    f_y_1_sum.append(sample)

                    sample_0 = torch.sum(risk[b,indices_j+1:],dim=0)
                    f_y_0_sum.append(sample_0)

                    sample_0_0 = torch.sum(risk[b,:],dim=0)
                    f_y_0_0_sum.append(sample_0_0)
                   
                f_y_1_sum = torch.stack(f_y_1_sum, dim=0)
                f_y_0_sum = torch.stack(f_y_0_sum, dim=0)
                f_y_0_0_sum = torch.stack(f_y_0_0_sum, dim=0)
                f_y_1 = torch.stack(f_y_1, dim=0)


                exp_theta = torch.exp(self.beta*(dis_t[i+1:])*(f_i_sum-f_y_1_sum[i+1:]))
                l_r = torch.max(1-exp_theta,torch.tensor(0).to(risk.device))

                
                l_un = f_i-torch.log(torch.sum(torch.exp(f_i_sum)))

                l_rc = label_state[i+1:]*(f_y_1[i+1:] - torch.log(torch.sum(torch.exp(f_y_1_sum[i+1:]))))+\
                        (1-label_state[i+1:])*(torch.log(torch.sum(f_y_0_sum[i+1:]))-torch.log(torch.sum(f_y_0_0_sum[i+1:])))


                loss = self.alpha_ij * l_r -self.alpha_i*l_un-self.alpha_j*l_rc

                tmc_loss.extend(loss)

        tmc_loss = torch.stack(tmc_loss, dim=0)
       
        return torch.mean(tmc_loss)