import torch.nn.functional as F
import torch
import numpy as np
from lifelines.utils import concordance_index # pip install lifelines

def c_index(risk_pred, y, e):
    ''' Performs calculating c-index
    shape:batch size

    :param risk_pred: (np.ndarray or torch.Tensor) model prediction(Negative numbers for risk prediction)
    :param y: (np.ndarray or torch.Tensor) the times of event e
    :param e: (np.ndarray or torch.Tensor) flag that records whether the event occurs
    :return c_index: the c_index is calculated by (risk_pred, y, e)
    '''
    risk_pred = F.sigmoid(risk_pred)
    if not isinstance(y, np.ndarray):
        y = y.detach().cpu().numpy()
    if not isinstance(risk_pred, np.ndarray):
        risk_pred = risk_pred.detach().cpu().numpy()
    if not isinstance(e, np.ndarray):
        e = e.detach().cpu().numpy()
    return concordance_index(y, risk_pred, e)



def weight_cindex_loss(predict, mtime, mstate, is_sigmoid=True):
    if is_sigmoid:
        predict = F.sigmoid(predict)
    index = torch.argsort(mtime, dim=0, descending=False)  # 在Batch Size 获取升序
    # 匹配升序的索引
    risk = torch.gather(input=predict, dim=0, index=index) # 根据OS的降序 改变predict 、oss 的顺序
    mstate = torch.gather(input=mstate, dim=0, index=index)

    cindex_loss = []
    for i, l in enumerate(mstate):
        if l == 1 and i<len(mstate)-1:
            dr = risk[i] - risk
            loss = torch.exp(-(dr[i:])/0.1).mean()
            cindex_loss.append(loss)

    return torch.stack(cindex_loss, dim=0).mean()



def balance_cindex_loss(predict, mtime, mstate, is_sigmoid=True):
    '''
    google的第二条损失函数
    '''
    if is_sigmoid:
        predict = F.sigmoid(predict)
    index = torch.argsort(mtime, dim=0, descending=False)  # 在Batch Size 获取升序
    # 匹配升序的索引
    risk = torch.gather(input=predict, dim=0, index=index) # 根据OS的降序 改变predict 、oss 的顺序
    mstate = torch.gather(input=mstate, dim=0, index=index)
    # 风险
    n = 0.
    loss = 0.

    for i, l in enumerate(mstate):
        if l == 1 and i<len(mstate)-1:
            dr = risk-risk[i]
            loss += torch.sum(1-torch.exp(dr[i+1:]))  # i+1不和自身比较
            n += len(dr[i+1:])
        
    loss = loss / n

    return -loss