# encoding: utf-8
"""
Training implementation for VIN-CXR Retrieval
Author: Jason.Fang
Update time: 22/03/2022
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
from resnet import resnet50
from vit import ViT
from vincxr_cls import get_box_dataloader_VIN
from pytorch_metric_learning import losses
from centroid_triplet_loss import CentroidTripletLoss
from cindex_triplet_loss import CIndexTripletLoss 

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
CLASS_NAMES = ['No finding', 'Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
                'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
CKPT_PATH = '/data/pycode/MedIR/CIndex/ckpts/vincxr_vit.pkl'
MAX_EPOCHS = 50
BATCH_SIZE = 16*8
#nohup python main_vincxr.py > logs/main_vincxr.log 2>&1 & 
def Train():
    print('********************load data********************')
    train_loader = get_box_dataloader_VIN(batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    #model = resnet50(pretrained=False, num_classes=len(CLASS_NAMES)*20).cuda()
    model = ViT(image_size = 224, patch_size = 32, num_classes = len(CLASS_NAMES)*20, dim = 1024, depth = 6,heads = 16, mlp_dim = 2048).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: "+CKPT_PATH)
    model = nn.DataParallel(model).cuda()
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
    #optional loss functions
    #criterion = losses.AngularLoss(alpha=40).cuda()
    #criterion = losses.ContrastiveLoss(pos_margin=0, neg_margin=1).cuda()
    #criterion = losses.TripletMarginLoss(margin=0.05, swap=False, smooth_loss=False, triplets_per_anchor="all").cuda()
    #criterion = CentroidTripletLoss(margin=0.05, swap=False, smooth_loss=False, triplets_per_anchor="all").cuda()
    #criterion = losses.SoftTripleLoss(num_classes=len(CLASS_NAMES), embedding_size=len(CLASS_NAMES)*20, centers_per_class=10, la=20, gamma=0.1, margin=0.01).cuda()
    #criterion = losses.CircleLoss(m=0.4, gamma=80).cuda()
    criterion = CIndexTripletLoss().cuda() #ours
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    loss_min = float('inf')
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        model.train()  #set model to training mode
        loss_train = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, label) in enumerate(train_loader):
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
        
        if loss_min > np.mean(loss_train):
            loss_min = np.mean(loss_train)
            torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch + 1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    train_loader = get_box_dataloader_VIN(batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_loader = get_box_dataloader_VIN(batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print ('==>>> total test batch number: {}'.format(len(test_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    #model = resnet50(pretrained=False, num_classes=len(CLASS_NAMES)*20).cuda()
    model = ViT(image_size = 224, patch_size = 32, num_classes = len(CLASS_NAMES)*20, dim = 1024, depth = 6,heads = 16, mlp_dim = 2048).cuda()
    criterion = CIndexTripletLoss().cuda() #ours
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
        for batch_idx, (image, label) in enumerate(train_loader):
            tr_label = torch.cat((tr_label, label.cuda()), 0)
            var_image = torch.autograd.Variable(image).cuda()
            var_feat = model(var_image)
            tr_feat = torch.cat((tr_feat, var_feat.data), 0)
            sys.stdout.write('\r train set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()

    te_label = torch.FloatTensor().cuda()
    te_feat = torch.FloatTensor().cuda()
    ci_score = []
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            te_label = torch.cat((te_label, label.cuda()), 0)
            var_image = torch.autograd.Variable(image).cuda()
            var_label = torch.autograd.Variable(label).cuda()
            var_feat = model(var_image)
            te_feat = torch.cat((te_feat, var_feat.data), 0)
            ci_score_batch = criterion.compute_CIScore(var_feat, var_label).item()
            if ci_score_batch >= 0.0: ci_score.append(ci_score_batch) #C-index metrci
            sys.stdout.write('\r test set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()

    print('********************Retrieval Performance!********************')
    sim_mat = cosine_similarity(te_feat.cpu().numpy(), tr_feat.cpu().numpy())
    te_label = te_label.cpu().numpy()
    tr_label = tr_label.cpu().numpy()

    for topk in [1, 5, 10]:
        mHRs_avg = []
        mAPs_avg = []
        for i in range(sim_mat.shape[0]):
            idxs, vals = zip(*heapq.nlargest(topk, enumerate(sim_mat[i,:].tolist()), key=lambda x:x[1]))
            num_pos = 0
            rank_pos = 0
            mAP = []
            for j in idxs:
                rank_pos = rank_pos + 1
                if te_label[i] == tr_label[j]:  #hit
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
        print("Average HR@{}={:.2f}".format(topk, np.mean(mHRs_avg)*100))
        #average precision
        print("Average AP@{}={:.2f}".format(topk, np.mean(mAPs_avg)*100))
    #C-Index Score
    print("Average CI={:.2f}".format(np.mean(ci_score)*100))

def main():
    Train() #for training
    Test() #for test

if __name__ == '__main__':
    main()
