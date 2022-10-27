# encoding: utf-8
"""
Training implementation of object detection for 2D chest x-ray
Author: Jason.Fang
Update time: 26/10/2022
"""
import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import pandas as pd
import torch
import math
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from thop import profile
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import cv2
import seaborn as sns
import torchvision.datasets as dset
import matplotlib.image as mpimg
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from nets.SDNet import SDNet

os.environ['CUDA_VISIBLE_DEVICES'] = "5,6"
def vis_weight():
    fig, axes = plt.subplots(1,2, constrained_layout=True,figsize=(12,6)) 

    model = SDNet(num_vectors=1000).cuda()
    for name, param in model.named_parameters():
        if "encoder.conv1" in name:
            pre_trained_on_imagenet = param.clone().cpu().data.numpy().flatten()
            pre_trained_on_imagenet =pre_trained_on_imagenet.flatten()
            break

    CKPT_PATH = '/data/pycode/MedIR/fundus/ckpts/ddr_resnet_cct_ln.pkl'
    model = SDNet(num_vectors=1000).cuda()
    checkpoint = torch.load(CKPT_PATH)
    model.load_state_dict(checkpoint) 
    for name, param in model.named_parameters():
        if "encoder.conv1" in name:
            cct_ln = param.clone().cpu().data.numpy().flatten()
            cct_ln = cct_ln.flatten()
            break
        
    CKPT_PATH = '/data/pycode/MedIR/fundus/ckpts/ddr_resnet_cct_sn.pkl'
    model = SDNet(num_vectors=1000).cuda()
    checkpoint = torch.load(CKPT_PATH)
    model.load_state_dict(checkpoint) 
    for name, param in model.named_parameters():
        if "encoder.conv1" in name:
            cct_sn = param.clone().cpu().data.numpy().flatten()
            cct_sn = cct_sn.flatten()
            break

    sns.distplot(pre_trained_on_imagenet, kde=True, ax=axes[0], hist_kws={'color':'green'}, kde_kws={'color':'green'}, label="Pre-train on ImageNet"+ r'$\ var=%.6f$' %(np.var(pre_trained_on_imagenet)))
    sns.distplot(cct_ln, kde=True, ax=axes[0], hist_kws={'color':'blue'}, kde_kws={'color':'blue'}, label="Trained by CCT with LNWD"+ r'$\ var=%.6f$' %(np.var(cct_ln)))
    sns.distplot(cct_sn, kde=True, ax=axes[0], hist_kws={'color':'red'}, kde_kws={'color':'red'}, label="Trained by CCT with SNWD"+ r'$\ var=%.6f$' %(np.var(cct_sn)))
    axes[0].grid(b=True, ls=':')
    axes[0].legend()

    CKPT_PATH = '/data/pycode/MedIR/fundus/ckpts/ddr_resnet_mt_ln.pkl'
    model = SDNet(num_vectors=1000).cuda()
    checkpoint = torch.load(CKPT_PATH)
    model.load_state_dict(checkpoint) 
    for name, param in model.named_parameters():
        if "encoder.conv1" in name:
            mt_ln = param.clone().cpu().data.numpy().flatten()
            mt_ln = mt_ln.flatten()
            break
        
    CKPT_PATH = '/data/pycode/MedIR/fundus/ckpts/ddr_resnet_mt_sn.pkl'
    model = SDNet(num_vectors=1000).cuda()
    checkpoint = torch.load(CKPT_PATH)
    model.load_state_dict(checkpoint) 
    for name, param in model.named_parameters():
        if "encoder.conv1" in name:
            mt_sn = param.clone().cpu().data.numpy().flatten()
            mt_sn = mt_sn.flatten()
            break

    sns.distplot(pre_trained_on_imagenet, kde=True, ax=axes[1], hist_kws={'color':'green'}, kde_kws={'color':'green'}, label="Pre-train on ImageNet: " + r'$\ var=%.6f$' %(np.var(pre_trained_on_imagenet)))
    sns.distplot(mt_sn, kde=True, ax=axes[1], hist_kws={'color':'blue'}, kde_kws={'color':'blue'}, label="Trained by MT with LNWD: "+ r'$\ var=%.6f$' %(np.var(mt_ln)))
    sns.distplot(mt_ln, kde=True, ax=axes[1], hist_kws={'color':'red'}, kde_kws={'color':'red'}, label="Trained by MT with SNWD: "+ r'$\ var=%.6f$' %(np.var(mt_sn)))
    axes[1].grid(b=True, ls=':')
    axes[1].legend()

    #output    
    fig.savefig('/data/pycode/MedIR/fundus/imgs/weight_hist.png', dpi=300, bbox_inches='tight')

def main():
    vis_weight()

if __name__ == '__main__':
    main()