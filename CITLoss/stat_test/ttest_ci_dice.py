import numpy as np
from scipy import stats
import pandas as pd


def dice_ci(dices, confidence=0.95):
    '''
    function: 通过有放回的重采样计算 CI
    :param y_true:  ground truth 
    :param y_score: 预测的概率
    '''
    sample_num = len(dices)
    df = sample_num - 1  # 自由度
    sample_mean = np.mean(dices)
    sample_sem = stats.sem(dices,ddof=0) # 标准误:sample_std/np.sqrt(len(dices))
   
    ci = stats.t.interval(confidence, df=df, loc=sample_mean, scale=sample_sem) # 获取均值置信区间

    return ci,sample_mean



if __name__ == '__main__':
    path = "/data/data_local_to_data/NPC_3D_code/val-result.csv"
    df = pd.read_csv(path)
    dices = df['dice']

    ci, sample_mean = dice_ci(dices, confidence=0.95)
    print('{:.3f}({:.3f}-{:.3f})'.format(sample_mean, ci[0],ci[1]))
