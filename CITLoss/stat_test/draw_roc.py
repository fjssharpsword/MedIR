from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, f1_score, precision_score, confusion_matrix


def draw_roc_curve(y_true, y_scores, save_path=None):
    """Draw roc curve.
    """
    # ap and auc, acc
    score_ap = average_precision_score(y_true, y_scores)
    auc_score = roc_auc_score(y_true, y_scores)
    print("Average precision: ", score_ap)
    print("AUC: ", auc_score)

    # draw roc curve.
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    index = np.argmax(tpr - fpr)
    # index = index+35
    # thresh = thresholds[index]
    thresh = 0.43
    print("thresh is ", thresh)

    y_scores[y_scores>=thresh] = 1
    y_scores[y_scores<thresh] = 0
    tn, fp, fn, tp = confusion_matrix(y_true, y_scores).ravel()
    recall = np.nan if (tp+fn) == 0 else float(tp)/(tp+fn)
    spe = np.nan if (tn+fp) == 0 else float(tn)/(tn+fp)

    print("recall", recall)
    print("spe", spe)
    print("precision", precision_score(y_true, y_scores))
    print("f1", f1_score(y_true, y_scores))

    if save_path is None:
        return

    print(zip(thresholds, tpr))
    plt.plot(fpr, tpr, lw=1, label='ROC curve(area = %0.3f)' % auc_score)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    for i in range(1, fpr.shape[0], 10):
        x = fpr[i]
        y = tpr[i]
        text= thresholds[i]
        plt.text(x, y+ 0.01, '%.3f' % text)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")

    # plt.show()
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    gt_path = '/data/code/deit/output/med2d/cxr_pretrainimgnet1k_1117/results_cvte_gt.csv'
    score_path = '/data/code/deit/output/med2d/cxr_pretrainimgnet1k_1117/results_cvte_pred.csv'
    save_path = "/data/code/deit/output/med2d/cxr_pretrainimgnet1k_1117/cvte.png"
    #gt
    gt_df = pd.read_csv(gt_path)
    # texture_lables = gt_df['Texture_label'].values
    gt_labels = gt_df['No finding'].values 

    #pred
    pred_df = pd.read_csv(score_path)
    preds = pred_df['No finding'].values

    draw_roc_curve(gt_labels, preds, save_path) #[texture_lables==0]


