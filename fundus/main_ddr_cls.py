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
from tensorboardX import SummaryWriter
#self-defined
from dsts.ddr_grading import get_fundus_DDR
from nets.SDNet import SDNet
from nets.CITLoss import CITLoss
from nets.WeightDecay import UpdateGrad

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "5,6"
CLASS_NAMES = ['No DR', "Mild NPDR", 'Moderate NPDR', 'Severe NPDR', 'PDR', 'Ungradable']
CKPT_PATH = '/data/pycode/MedIR/fundus/ckpts/ddr_resnet_sd_cit.pkl'
MAX_EPOCHS = 20

def Train():
    print('********************load data********************')
    dataloader_train = get_fundus_DDR(batch_size=200, shuffle=True, num_workers=8, dst_type='train')
    dataloader_valid = get_fundus_DDR(batch_size=200, shuffle=False, num_workers=8, dst_type='valid')
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
    #criterion = losses.TripletMarginLoss(margin=0.05, swap=False, smooth_loss=False, triplets_per_anchor="all").cuda()
    #criterion = losses.CircleLoss(m=0.4, gamma=80).cuda()
    criterion = CITLoss(gamma = 0.5).cuda()#1.0-0.5-0.1
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    log_writer = SummaryWriter('/data/tmpexec/tb_log')
    loss_max = float('inf')
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
                UpdateGrad(model, coef=1e-3, p='sd') #weight decay, p=sd/l2
                optimizer.step()
                #show 
                loss_train.append(loss_bn.item())
                sys.stdout.write('\r Train Epoch: {} / Step: {} : loss = {}'.format(epoch+1, batch_idx+1, float('%0.4f'%loss_bn.item()) ))
                sys.stdout.flush()
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Train Eopch: %5d loss = %.4f " % (epoch + 1, np.mean(loss_train) ))

        model.eval()
        loss_val = []
        with torch.autograd.no_grad():
            for batch_idx,  (img, lbl) in enumerate(dataloader_valid):
                var_img = torch.autograd.Variable(img).cuda()
                var_lbl = torch.autograd.Variable(lbl).cuda()
                var_vec = model(var_img)
                loss_bn = criterion.forward(var_vec, var_lbl)
                loss_val.append(loss_bn.item())

                sys.stdout.write('\r Valid Epoch: {} / Step: {} : loss = {}'.format(epoch+1, batch_idx+1, float('%0.4f'%loss_bn.item()) ))
                sys.stdout.flush()
        print("\r Valid Eopch: %5d loss = %.4f" % (epoch + 1, np.mean(loss_val) ))

        if loss_max > np.mean(loss_val):
            loss_max = np.mean(loss_val)
            torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch + 1))

        for name, param in model.named_parameters():
            if "conv" in name:
                log_writer.add_histogram(name + '_data', param.clone().cpu().data.numpy(), epoch+1)
                if param.grad is not None: #leaf node in the graph retain gradient
                    log_writer.add_histogram(name + '_grad', param.grad, epoch+1)

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))
        log_writer.add_scalars('DDR_Resnet/SD_CITLoss', {'Train':np.mean(loss_train), 'Test':np.mean(loss_val)}, epoch+1)

    log_writer.close() #shut up the tensorboard

def Test():
    print('********************load data********************')
    dataloader_train = get_fundus_DDR(batch_size=100, shuffle=False, num_workers=6, dst_type='train')
    dataloader_valid = get_fundus_DDR(batch_size=100, shuffle=False, num_workers=6, dst_type='valid')
    dataloader_test = get_fundus_DDR(batch_size=100, shuffle=False, num_workers=6, dst_type='test')
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = SDNet(num_vectors=1000).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: "+CKPT_PATH)
    model.eval()#turn to test mode
    print('******************** load model succeed!********************')

    print('********************Build gallery********************')
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
    print('********************Build gallery succeed!********************')

    print('********************Build query********************')
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
    print('********************Build query succeed!********************')

    print('********************Retrieval Performance!********************')
    sim_mat = cosine_similarity(te_vec.cpu().numpy(), db_vec.cpu().numpy())
    te_lbl = te_lbl.cpu().numpy()
    db_lbl = db_lbl.cpu().numpy()

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
                if te_lbl[i] == db_lbl[j]:  #hit
                    num_pos = num_pos +1
                    mAP.append(num_pos/rank_pos)
                else:
                    mAP.append(0)
            if len(mAP) > 0:
                mAPs_avg.append(np.mean(mAP))
            else:
                mAPs_avg.append(0)
            mHRs_avg.append(num_pos/rank_pos)

            sys.stdout.write('\r query process: = {}'.format(i+1))
            sys.stdout.flush()

        #Hit ratio
        print("\r HR@{}={:.4f}".format(topk, np.mean(mHRs_avg)))
        #average precision
        print("\r AP@{}={:.4f}".format(topk, np.mean(mAPs_avg)))

def main():
    Train() #for training
    Test() #for test

if __name__ == '__main__':
    main()
    #nohup python3 main_ddr_cls.py > logs/main_ddr_cls.log 2>&1 &
