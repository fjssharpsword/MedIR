import math
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd


def auc_ci(y_true, y_score, thresh=0.5, confidence=0.95):
    '''
    function: 通过有放回的重采样计算 CI
    :param y_true:  ground truth 
    :param y_score: 预测的概率
    '''
    n = len(y_true)  # 抽样的样本数量
    N = 1000  # 抽样次数
    sample_roc_auc = []
    sample_sen = []
    sample_spe = []
    for k in range(N):
        sample = np.random.choice(n,n,replace=True) # 有放回抽样[0,n)
        sample_y_true = []
        sample_y_score = []
        for i in sample:  # 抽出的样本
            sample_y_true.append(y_true[i])
            sample_y_score.append(y_score[i])
        sample_fpr, sample_tpr, _ = roc_curve(sample_y_true, sample_y_score)
        sample_roc_auc.append(auc(sample_fpr, sample_tpr))

        sample_y_score = np.array(sample_y_score)
        sample_y_score[sample_y_score>=thresh] = 1
        sample_y_score[sample_y_score<thresh] = 0
        tn, fp, fn, tp = confusion_matrix(sample_y_true, sample_y_score).ravel()
        sen = np.nan if (tp+fn) == 0 else float(tp)/(tp+fn)
        spe = np.nan if (tn+fp) == 0 else float(tn)/(tn+fp)
        sample_sen.append(sen)
        sample_spe.append(spe)
        
    mean_auc, lb_auc, up_auc = compute_ci(sample_roc_auc, confidence)
    mean_sen, lb_sen, up_sen = compute_ci(sample_sen, confidence)
    mean_spe, lb_spe, up_spe = compute_ci(sample_spe, confidence)
    
    print('{:.3f}({:.3f}-{:.3f})'.format(mean_auc, lb_auc, up_auc))
    print('{:.3f}({:.3f}-{:.3f})'.format(mean_sen, lb_sen, up_sen))
    print('{:.3f}({:.3f}-{:.3f})'.format(mean_spe, lb_spe, up_spe))


def compute_ci(boot_samples, confidence):
    sample_num = len(boot_samples)
    Pz = (1.0-confidence)/2.0
    boot_samples.sort()

    mean = np.average(boot_samples)
    lb = boot_samples[int(math.floor(Pz*sample_num))]
    up = boot_samples[int(math.floor((1.0-Pz)*sample_num))]
    return mean, lb, up


if __name__ == '__main__':
    path = "/data/Metrics/temp/nlst_once_cnn.csv"
    df = pd.read_csv(path)
    y_true, y_score = df['label'], df['pred']
    auc_ci(y_true, y_score, thresh=0.43, confidence=0.95)