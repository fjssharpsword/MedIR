# encoding: utf-8
"""
Training implementation for MRE
Author: Jason.Fang
Update time: 23/12/2021
"""
import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from skimage.measure import label
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics import ndcg_score
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, confusion_matrix, accuracy_score
#self-defined
from nets.mre import MRE
from dsts.idrid_grading import get_train_dataset_fundus, get_test_dataset_fundus


#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
CLASS_NAMES = ['Normal', "Mild NPDR", 'Moderate NPDR', 'Severe NPDR', 'PDR']
CKPT_PATH = '/data/pycode/MedIR/ckpts/mre_fundus.pkl'
MAX_EPOCHS = 20

def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataset_fundus(batch_size=16, shuffle=True, num_workers=1)
    dataloader_test = get_test_dataset_fundus(batch_size=16, shuffle=False, num_workers=1)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    model = MRE(num_classes=5).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: "+CKPT_PATH)
    model = nn.DataParallel(model).cuda()
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
    #define loss function
    criterion = nn.BCELoss().cuda() #nn.CrossEntropyLoss().cuda() 
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    #loss_min = float('inf')
    AUROC_best = 0.50
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        model.train()  #set model to training mode
        loss_train = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, label) in enumerate(dataloader_train):
                optimizer.zero_grad()
                #forward
                var_image = torch.autograd.Variable(image).cuda()
                var_label = torch.autograd.Variable(label).cuda()
                var_output = model(var_image)
                # backward and update parameters
                loss_tensor = criterion.forward(var_output, var_label)
                loss_tensor.backward()
                optimizer.step()
                #show 
                loss_train.append(loss_tensor.item())
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(loss_train) ))

        model.eval()#turn to test mode
        gt = torch.FloatTensor()
        pred = torch.FloatTensor()
        with torch.autograd.no_grad():
            for batch_idx, (image, label) in enumerate(dataloader_test):
                var_image = torch.autograd.Variable(image).cuda()
                var_output = model(var_image)#forward
                gt = torch.cat((gt, label), 0)
                pred = torch.cat((pred, var_output.data.cpu()), 0)
                sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
                sys.stdout.flush()
        gt_np = gt.numpy()
        pred_np = pred.numpy() 
        AUROCs = []
        for i in range(len(CLASS_NAMES)):
            AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
        print("\r Epoch: %5d Validation average AUROC = %.2f" % ( epoch + 1, np.mean(AUROCs)*100 ) )

        if AUROC_best < np.mean(AUROCs):
            AUROC_best = np.mean(AUROCs)
            torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch + 1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    dataloader_train = get_train_dataset_fundus(batch_size=8, shuffle=True, num_workers=1)
    dataloader_test = get_test_dataset_fundus(batch_size=8, shuffle=True, num_workers=1)
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = MRE(num_classes=5).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: "+CKPT_PATH)
    model.eval()#turn to test mode
    print('******************** load model succeed!********************')

    print('********************Build feature database!********************')
    tr_label = torch.FloatTensor().cuda()
    tr_feat = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_train):
            tr_label = torch.cat((tr_label, label.cuda()), 0)
            var_image = torch.autograd.Variable(image).cuda()
            var_feat = model(var_image)
            tr_feat = torch.cat((tr_feat, var_feat.data), 0)
            sys.stdout.write('\r train set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()

    te_label = torch.FloatTensor().cuda()
    te_feat = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_test):
            te_label = torch.cat((te_label, label.cuda()), 0)
            var_image = torch.autograd.Variable(image).cuda()
            var_feat = model(var_image)
            te_feat = torch.cat((te_feat, var_feat.data), 0)
            sys.stdout.write('\r test set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()

    print('********************Retrieval Performance!********************')
    sim_mat = cosine_similarity(te_feat.cpu().numpy(), tr_feat.cpu().numpy())
    te_label = te_label.cpu().numpy()
    tr_label = tr_label.cpu().numpy()

    for topk in [5,10,20,50]:
        mHRs_avg = []
        mAPs_avg = []
        for i in range(sim_mat.shape[0]):
            idxs, vals = zip(*heapq.nlargest(topk, enumerate(sim_mat[i,:].tolist()), key=lambda x:x[1]))
            num_pos = 0
            rank_pos = 0
            mAP = []
            te_idx = te_label[i,:][0]
            for j in idxs:
                rank_pos = rank_pos + 1
                tr_idx = tr_label[j,:][0]
                if abs(tr_idx - te_idx) < np.mean(tr_label):  #hit
                    num_pos = num_pos +1
                    mAP.append(num_pos/rank_pos)
                else:
                    mAP.append(0)
            if len(mAP) > 0:
                mAPs_avg.append(np.mean(mAP))
            else:
                mAPs_avg.append(0)
            mHRs_avg.append(num_pos/rank_pos)

            sys.stdout.write('\r test set process: = {}'.format(i+1))
            sys.stdout.flush()

        #Hit ratio
        logger.info("MRE Average mHR@{}={:.4f}".format(topk, np.mean(mHRs_avg)))
        #average precision
        logger.info("MRE Average mAP@{}={:.4f}".format(topk, np.mean(mAPs_avg)))


def main():
    Train() #for training
    Test() #for test

if __name__ == '__main__':
    main()
    #nohup python main_mre_fundus.py > logs/train.log 2>&1 &
