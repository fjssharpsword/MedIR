import torch
from pycox.models.utils import pad_col
import torch.nn as nn

class CELoss(object):

    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.celoss = nn.CrossEntropyLoss().cuda()
        

    def __call__(self, results_dict, survival_time,state_label,interval_label): 
        # loss = self.celoss(results_dict['logits'],interval_label.long())
        return ce_loss(results_dict, survival_time, state_label,interval_label)# + loss
        # return loss


def ce_loss(results_dict, survival_time,state_label,interval_label, eps=1e-7):
    # w = torch.tensor([0.27,0.32,0.22,0.05])
    hazards = results_dict['logits']
    hazards = pad_col(hazards).softmax(1)
    # .softmax(1)
    batch_size = len(survival_time)
    # w = w.repeat(batch_size,1)
    surv_time = survival_time.view(batch_size, 1) # ground truth bin, 1,2,...,k
    sta_label = state_label.view(batch_size, 1).float() #censorship status, 0 or 1
    inter_label = interval_label.view(batch_size, 1)

    
    # death = (1 - censoring) * (torch.log(torch.gather(hazards, 1, survival_time.long()).clamp(min=eps)))
    death = sta_label * torch.log(torch.gather(hazards, 1, inter_label.long()))

    cumsum = hazards.cumsum(1)
    censor = torch.log(1-torch.gather(cumsum, 1, inter_label.long()))

    # censor = []
    # for b in range(batch_size):
    #     indices = interval_label[b].long()+1
    #     sample = torch.sum(torch.log(hazards[b,indices:]))
    #     censor.append(sample)
    # censor = torch.stack(censor, dim=0)

    censor = (1-sta_label) * censor
    loss = - (death + censor)        #* torch.gather(w.cuda(), 1, inter_label.long())
    loss = loss.mean()
    
    return loss