# encoding: utf-8
"""
Training implementation for DDR grading dataset.
Author: Jason.Fang
Update time: 09/10/2022
"""
import sys
import os
import time
import heapq
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from pytorch_metric_learning import losses
from sklearn.cluster import KMeans
from sklearn import metrics
from tensorboardX import SummaryWriter
#self-defined
from dsts.ddr_grading import get_fundus_DDR
from dsts.idrid_grading import get_fundus_idrid
from nets.SDNet import SDNet
from nets.CITLoss import CITLoss
from nets.WeightDecay import UpdateGrad

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "5,6"
CLASS_NAMES = ['No DR', "Mild NPDR", 'Moderate NPDR', 'Severe NPDR', 'PDR']
CKPT_PATH = '/data/pycode/MedIR/fundus/ckpts/ddr_resnet_sn_cit.pkl'
MAX_EPOCHS = 30

def Train():
    print('********************load data********************')
    dataloader_train = get_fundus_DDR(batch_size=128, shuffle=True, num_workers=8, dst_type='train')
    dataloader_valid = get_fundus_DDR(batch_size=128, shuffle=False, num_workers=8, dst_type='valid')
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    model = SDNet(num_vectors=1000).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: "+CKPT_PATH)
    model = nn.DataParallel(model).cuda()
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0) #default:0.0, L2 = 1e-3
    lr_scheduler_model = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
    kmeans = KMeans(n_clusters=len(CLASS_NAMES), random_state=0)
    #criterion = losses.TripletMarginLoss(margin=0.05, swap=False, smooth_loss=False, triplets_per_anchor="all").cuda()
    #criterion = losses.CircleLoss(m=0.4, gamma=80).cuda()
    criterion = CITLoss(gamma = 0.5).cuda()#1.0-0.5-0.1
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    log_writer = SummaryWriter('/data/tmpexec/tb_log')
    #loss_max = float('inf')
    FMI_min = 0.0
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        model.train()  #set model to training mode
        loss_train= []
        with torch.autograd.enable_grad():
            for batch_idx, (img, lbl) in enumerate(dataloader_train):
                optimizer.zero_grad()
                #forward
                var_img = torch.autograd.Variable(img).cuda()
                var_lbl = torch.autograd.Variable(lbl).cuda()
                var_vec = model(var_img)
                # backward and update parameters
                loss_bn = criterion.forward(var_vec, var_lbl)
                loss_bn.backward()
                #sn-spectral norm-coef=1.0, l2n-l2norm-coef=0.1/0.01
                UpdateGrad(model, coef=1.0, p='sn') 
                #UpdateGrad(model, coef=0.1, p='ln') 
                optimizer.step()
                #show 
                loss_train.append(loss_bn.item())
                sys.stdout.write('\r Train Epoch: {} / Step: {} : loss = {}'.format(epoch+1, batch_idx+1, float('%0.4f'%loss_bn.item()) ))
                sys.stdout.flush()
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Train Eopch: %5d loss = %.4f " % (epoch + 1, np.mean(loss_train) ))

        model.eval()
        loss_val = []
        val_lbl = torch.FloatTensor().cuda()
        val_vec = torch.FloatTensor().cuda()
        with torch.autograd.no_grad():
            for batch_idx,  (img, lbl) in enumerate(dataloader_valid):
                var_img = torch.autograd.Variable(img).cuda()
                var_lbl = torch.autograd.Variable(lbl).cuda()
                var_vec = model(var_img)
                loss_bn = criterion.forward(var_vec, var_lbl)
                loss_val.append(loss_bn.item())

                val_vec = torch.cat((val_vec, var_vec.data), 0)
                val_lbl = torch.cat((val_lbl, lbl.cuda()), 0)

                sys.stdout.write('\r Valid Epoch: {} / Step: {} : loss = {}'.format(epoch+1, batch_idx+1, float('%0.4f'%loss_bn.item()) ))
                sys.stdout.flush()
        print("\r Valid Eopch: %5d loss = %.4f" % (epoch + 1, np.mean(loss_val) ))
        pre_lbl = kmeans.fit(val_vec.cpu().numpy()).labels_
        FMI = metrics.fowlkes_mallows_score(val_lbl.cpu().numpy(), pre_lbl)
        print("\r Valid Eopch: %5d Fowlkes-Mallows Index (FMI) = %.4f" % (epoch + 1, FMI))

        if FMI_min < FMI:
            FMI_min = FMI
        #if loss_max > np.mean(loss_train):
        #    loss_max = np.mean(loss_train)
            torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch + 1))

        if (epoch+1)==1 or (epoch+1) == MAX_EPOCHS:
            for name, param in model.named_parameters():
                if "conv" in name:
                    log_writer.add_histogram(name + '_data', param.clone().cpu().data.numpy(), epoch+1)
                    if param.grad is not None: #leaf node in the graph retain gradient
                        log_writer.add_histogram(name + '_grad', param.grad, epoch+1)

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))
        log_writer.add_scalars('DDR_Resnet/SN_Triplet', {'Train':np.mean(loss_train), 'Test':np.mean(loss_val), 'FMI':FMI}, epoch+1)

    log_writer.close() #shut up the tensorboard

def Query():
    print('\r ********************load data********************')
    dataloader_train = get_fundus_DDR(batch_size=64, shuffle=False, num_workers=4, dst_type='train')
    dataloader_valid = get_fundus_DDR(batch_size=64, shuffle=False, num_workers=4, dst_type='valid')
    dataloader_test = get_fundus_DDR(batch_size=64, shuffle=False, num_workers=4, dst_type='test')
    dataloader_idrid = get_fundus_idrid(batch_size=64, shuffle=False, num_workers=4)
    print('\r ********************load data succeed!********************')

    print('\r ********************load model********************')
    model = SDNet(num_vectors=1000).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: "+CKPT_PATH)
    model.eval()#turn to test mode
    kmeans = KMeans(n_clusters=len(CLASS_NAMES), random_state=0)
    print('\r ******************** load model succeed!********************')

    print('\r ********************Build gallery********************')
    db_lbl = torch.FloatTensor().cuda()
    db_vec = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (img, lbl) in enumerate(dataloader_train):
            db_lbl = torch.cat((db_lbl, lbl.cuda()), 0)
            var_img = torch.autograd.Variable(img).cuda()
            var_vec = model(var_img)
            db_vec = torch.cat((db_vec, var_vec.data), 0)
            sys.stdout.write('\r train set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()
        for batch_idx, (img, lbl) in enumerate(dataloader_valid):
            db_lbl = torch.cat((db_lbl, lbl.cuda()), 0)
            var_img = torch.autograd.Variable(img).cuda()
            var_vec = model(var_img)
            db_vec = torch.cat((db_vec, var_vec.data), 0)
            sys.stdout.write('\r valid set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()
    print('\r ********************Build gallery succeed!********************')

    print('\r ********************Build query for DDR dataset********************')
    te_lbl = torch.FloatTensor().cuda()
    te_vec = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (img, lbl) in enumerate(dataloader_test):
            te_lbl = torch.cat((te_lbl, lbl.cuda()), 0)
            var_img = torch.autograd.Variable(img).cuda()
            var_vec = model(var_img)
            te_vec = torch.cat((te_vec, var_vec.data), 0)
            sys.stdout.write('\r test set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()
    #metric
    sim_mat = cosine_similarity(te_vec.cpu().numpy(), db_vec.cpu().numpy())
    te_lbl = te_lbl.cpu().numpy()
    db_lbl = db_lbl.cpu().numpy()

    #calcuate Hit ratio
    HRs = {0:[], 1:[], 2:[], 3:[], 4:[]}
    mHR = []
    for i in range(sim_mat.shape[0]):
        idxs, vals = zip(*heapq.nlargest(1, enumerate(sim_mat[i,:].tolist()), key=lambda x:x[1]))#top1
        j = idxs[0]
        if te_lbl[i] == db_lbl[j]:  #hit
            mHR.append(1)
            HRs[te_lbl[i]].append(1)
        else:
            mHR.append(0)
            HRs[te_lbl[i]].append(0)
    #Hit ratio
    for i in range(len(CLASS_NAMES)):
        print ("\r {}: HR@{}={:.2f}".format(CLASS_NAMES[i], 1, np.mean(HRs[i])*100))
    print("\r mHR@{}={:.2f}".format(1, np.mean(mHR)*100))

    #calcuate average precision
    for topk in [5, 10]:
        APs = {0:[], 1:[], 2:[], 3:[], 4:[]}
        mAP = []
        for i in range(sim_mat.shape[0]):
            idxs, vals = zip(*heapq.nlargest(topk, enumerate(sim_mat[i,:].tolist()), key=lambda x:x[1]))
            num_pos = 0
            rank_pos = 0
            AP = []
            for j in idxs:
                rank_pos = rank_pos + 1
                if te_lbl[i] == db_lbl[j]:  #hit
                    num_pos = num_pos + 1
                    AP.append(num_pos/rank_pos)
                #else: AP.append(0)
            if len(AP) > 0:
                mAP.append(np.mean(AP))
                APs[te_lbl[i]].append(np.mean(AP))
            else:
                mAP.append(0)
                APs[te_lbl[i]].append(0)
        for i in range(len(CLASS_NAMES)):
            print ("\r {}: AP@{}={:.2f}".format(CLASS_NAMES[i], topk, np.mean(APs[i])*100))
        print("\r mAP@{}={:.2f}".format(topk, np.mean(mAP)*100))

    #clustering performance evaluation
    pre_lbl = kmeans.fit(te_vec.cpu().numpy()).labels_
    aRI = metrics.adjusted_rand_score(te_lbl, pre_lbl)
    print("\r adjusted Rand Index (aRI)={:.2f}".format(aRI*100))
    aMI = metrics.adjusted_mutual_info_score(te_lbl, pre_lbl)
    print("\r adjusted Mutual Information (aMI) ={:.2f}".format(aMI*100))
    FMI = metrics.fowlkes_mallows_score(te_lbl, pre_lbl)
    print("\r Fowlkes-Mallows Index (FMI) ={:.2f}".format(FMI*100))
    print('\r ********************Build query for DDR dataset succeed!********************')

    print('\r ********************Build query for IDRiD dataset********************')
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
    #metric
    sim_mat = cosine_similarity(te_vec.cpu().numpy(), db_vec.cpu().numpy())
    te_lbl = te_lbl.cpu().numpy()
    #db_lbl = db_lbl.cpu().numpy()

    #calcuate Hit ratio
    HRs = {0:[], 1:[], 2:[], 3:[], 4:[]}
    mHR = []
    for i in range(sim_mat.shape[0]):
        idxs, vals = zip(*heapq.nlargest(1, enumerate(sim_mat[i,:].tolist()), key=lambda x:x[1]))#top1
        j = idxs[0]
        if te_lbl[i] == db_lbl[j]:  #hit
            mHR.append(1)
            HRs[te_lbl[i]].append(1)
        else:
            mHR.append(0)
            HRs[te_lbl[i]].append(0)
    #Hit ratio
    for i in range(len(CLASS_NAMES)):
        print ("\r {}: HR@{}={:.2f}".format(CLASS_NAMES[i], 1, np.mean(HRs[i])*100))
    print("\r mHR@{}={:.2f}".format(1, np.mean(mHR)*100))

    #calcuate average precision
    for topk in [5, 10]:
        APs = {0:[], 1:[], 2:[], 3:[], 4:[]}
        mAP = []
        for i in range(sim_mat.shape[0]):
            idxs, vals = zip(*heapq.nlargest(topk, enumerate(sim_mat[i,:].tolist()), key=lambda x:x[1]))
            num_pos = 0
            rank_pos = 0
            AP = []
            for j in idxs:
                rank_pos = rank_pos + 1
                if te_lbl[i] == db_lbl[j]:  #hit
                    num_pos = num_pos + 1
                    AP.append(num_pos/rank_pos)
                #else: AP.append(0)
            if len(AP) > 0:
                mAP.append(np.mean(AP))
                APs[te_lbl[i]].append(np.mean(AP))
            else:
                mAP.append(0)
                APs[te_lbl[i]].append(0)
        for i in range(len(CLASS_NAMES)):
            print ("\r {}: AP@{}={:.2f}".format(CLASS_NAMES[i], topk, np.mean(APs[i])*100))
        print("\r mAP@{}={:.2f}".format(topk, np.mean(mAP)*100))

    #clustering performance evaluation
    pre_lbl = kmeans.fit(te_vec.cpu().numpy()).labels_
    aRI = metrics.adjusted_rand_score(te_lbl, pre_lbl)
    print("\r adjusted Rand Index (aRI)={:.2f}".format(aRI*100))
    aMI = metrics.adjusted_mutual_info_score(te_lbl, pre_lbl)
    print("\r adjusted Mutual Information (aMI) ={:.2f}".format(aMI*100))
    FMI = metrics.fowlkes_mallows_score(te_lbl, pre_lbl)
    print("\r Fowlkes-Mallows Index (FMI) ={:.2f}".format(FMI*100))
    print('\r ********************Build query for IDRiD dataset succeed!********************')
    
def main():
    Train() #for training
    Query() #for test

if __name__ == '__main__':
    main()
    #nohup python3 main_ddr_cls.py > logs/main_ddr_cls.log 2>&1 &
