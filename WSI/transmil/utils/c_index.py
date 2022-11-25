import torch
from lifelines.utils import concordance_index
import numpy as np
import numba

def is_concordant_func(s_i,s_j,t_i,t_j,d_i,d_j):

    return (s_i < s_j) & is_comparable_func(t_i,t_j,d_i,d_j)

def is_comparable_func(t_i,t_j,d_i,d_j):
    return ((t_i < t_j) & d_i) | ((t_i == t_j) & d_i & (d_j == 0))

def my_pycox_cindex(surv,total_time,total_label):
    n = total_time.shape[0]
    s_idx = np.searchsorted([0,1,2,3], total_time, side='right') - 1
    count_z = 0
    for i in numba.prange(n):
        idx = s_idx[i]
        for j in range(n):
            if j != i:
                count_z += is_concordant_func(surv[i,idx], surv[j,idx], total_time[i].item(), total_time[j].item(), int(total_label[i].item()), int(total_label[j].item()))


    n = total_time.shape[0]
    count_m = 0.
    for i in numba.prange(n):
        for j in range(n):
            if j != i:
                count_m += is_comparable_func(total_time[i].item(), total_time[j].item(), int(total_label[i].item()), int(total_label[j].item()))

    return count_z / count_m

def ours_cindex(surv,total_time,total_label,s_idx):
    n = total_time.shape[0]
    # s_idx = np.searchsorted([0,1,2,3], total_time, side='right') - 1
    count_z = 0
    for i in numba.prange(n):
        idx = int(s_idx[i].item())
        for j in range(n):
            if j != i:
                count_z += is_concordant_func(surv[i,idx], surv[j,idx], total_time[i].item(), total_time[j].item(), int(total_label[i].item()), int(total_label[j].item()))


    n = total_time.shape[0]
    count_m = 0.
    for i in numba.prange(n):
        for j in range(n):
            if j != i:
                count_m += is_comparable_func(total_time[i].item(), total_time[j].item(), int(total_label[i].item()), int(total_label[j].item()))

    return count_z / count_m

    
def c_index(predict, survival_labels, censorship_label):
    """

    :param predict: 预测的生存信息
    :param survival_labels: 真实的生存信息
    :return:
    """
    # predict = predict.view(-1)
    # survival_values, index = torch.sort(survival_labels, dim=0, descending=True)
    # predict_values = torch.gather(input=predict, dim=0, index=index).cpu().detach().numpy()
    # censorship_values = torch.gather(input=censorship_label, dim=0, index=index).cpu().detach().numpy()
    predict_values = predict.cpu().detach().numpy()
    survival_labels = survival_labels.cpu().detach().numpy()
    censorship_label = censorship_label.cpu().detach().numpy()

    if not np.any(censorship_label):
        score = None
    else:
        score = concordance_index(event_times=survival_labels,
                                  predicted_scores=predict_values,
                                  event_observed=censorship_label)
    return score


if __name__ == '__main__':
    predict = torch.tensor([
        [0.2032],
        [0.0643],
        [0.1554],
        [-0.0061],
        [0.0344],
        [0.0784],
        [-0.0088],
        [0.1577]], dtype=torch.float16)
    label_os = torch.tensor([[684.],
                             [2820.],
                             [826.],
                             [4601.],
                             [160.],
                             [188.],
                             [383.],
                             [578.]], dtype=torch.float64)

    label_oss = torch.tensor([[1],
                              [1],
                              [1],
                              [1],
                              [1],
                              [1],
                              [1],
                              [1]])
    # label_oss = torch.tensor([[0],
    #                           [0],
    #                           [0],
    #                           [0],
    #                           [0],
    #                           [0],
    #                           [0],
    #                           [0]])

    score = c_index(predict, label_os, label_oss)
    print(score)
