import torch
from pycox.models.utils import pad_col
from utils.losses.ce_loss import CELoss

class CEPlusLoss(object):

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.ce_loss = CELoss()

    def __call__(self, results_dict, survival_time,state_label,interval_label): 

        celoss = self.ce_loss(results_dict, survival_time, state_label,interval_label)
        ceplusloss = ce_plus_loss(results_dict, survival_time, state_label,interval_label)

        loss = (1 - self.alpha) * celoss + self.alpha * ceplusloss

        return loss



def ce_plus_loss(results_dict, survival_time, state_label,interval_label, alpha=0.4, eps=1e-7):
    hazards = results_dict['logits']
    hazards = pad_col(hazards).softmax(1)

    batch_size = len(survival_time)
    surv_time = survival_time.view(batch_size, 1) # ground truth bin, 1,2,...,k
    sta_label = state_label.view(batch_size, 1).float() #censorship status, 0 or 1
    inter_label = interval_label.view(batch_size, 1)


    death = sta_label * torch.log(torch.gather(hazards, 1, inter_label.long()))

    censor = []
    for b in range(batch_size):
        indices = interval_label[b].long()
        sample = torch.sum(1-hazards[b,:indices])
        censor.append(sample)
    censor = torch.stack(censor, dim=0)

    censor = (1-state_label) * censor
    loss =  - (death.view(-1) + censor)
    loss = loss.mean()
    return loss