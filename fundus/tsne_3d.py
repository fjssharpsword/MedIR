# encoding: utf-8
"""
Training implementation of object detection for 2D chest x-ray
Author: Jason.Fang
Update time: 26/10/2022
"""
import sys
import os
import numpy as np
import pandas as pd
import torch
from thop import profile
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from nets.SDNet import SDNet
from dsts.idrid_grading import get_fundus_idrid
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

os.environ['CUDA_VISIBLE_DEVICES'] = "5,6"
def tsne_3d():
    dataloader_idrid = get_fundus_idrid(batch_size=64, shuffle=False, num_workers=2)
    """
    CKPT_PATH = '/data/pycode/MedIR/fundus/ckpts/ddr_resnet_mt_ln.pkl'
    model = SDNet(num_vectors=1000).cuda()
    checkpoint = torch.load(CKPT_PATH)
    model.load_state_dict(checkpoint) 
    te_lbl = torch.FloatTensor().cuda()
    te_vec = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (img, lbl) in enumerate(dataloader_idrid):
            te_lbl = torch.cat((te_lbl, lbl.cuda()), 0)
            var_img = torch.autograd.Variable(img).cuda()
            var_vec = model(var_img)
            te_vec = torch.cat((te_vec, var_vec.data), 0)
            sys.stdout.write('\r test set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()
    te_vec_mt_ln = te_vec.cpu().numpy()
    tsne_mt_ln = TSNE(n_components=3).fit_transform(te_vec_mt_ln)
    """

    CKPT_PATH = '/data/pycode/MedIR/fundus/ckpts/ddr_resnet_cct_sn.pkl'
    model = SDNet(num_vectors=1000).cuda()
    checkpoint = torch.load(CKPT_PATH)
    model.load_state_dict(checkpoint) 
    te_lbl = torch.FloatTensor().cuda()
    te_vec = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (img, lbl) in enumerate(dataloader_idrid):
            te_lbl = torch.cat((te_lbl, lbl.cuda()), 0)
            var_img = torch.autograd.Variable(img).cuda()
            var_vec = model(var_img)
            te_vec = torch.cat((te_vec, var_vec.data), 0)
            sys.stdout.write('\r test set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()
    te_vec_cct_sn = te_vec.cpu().numpy()

    tsne_cct_sn = TSNE(n_components=3,perplexity=100).fit_transform(te_vec_cct_sn)
    fig, axes = plt.subplots(1) 
    #axes= Axes3D(fig)
    #axes.scatter(tsne_cct_sn[:,0], tsne_cct_sn[:, 1], tsne_cct_sn[:, 2],s=2,c=te_lbl.cpu().numpy()) 
    axes.scatter(tsne_cct_sn[:,0], tsne_cct_sn[:, 1], c=te_lbl.cpu().numpy(), s=2, alpha=0.5)
    #output    
    fig.savefig('/data/pycode/MedIR/fundus/imgs/tsne_2d.png', dpi=300, bbox_inches='tight')

def main():
    tsne_3d()

if __name__ == '__main__':
    main()