# encoding: utf-8
"""
Training implementation for DDR grading dataset.
Author: Jason.Fang
Update time: 17/10/2022
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
from sklearn import metrics
from sklearn.metrics import ndcg_score
from tensorboardX import SummaryWriter
#self-defined
from dsts.idrid_seg import get_train_dataloader, get_test_dataloader
from nets.vae import BetaVAE
from nets.unet_2d import UNet

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "5,6"
CKPT_PATH = '/data/pycode/MedIR/fundus/ckpts/idrid_vae_grad.pkl'
MAX_EPOCHS = 200

def Train(lesion='MA'):
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=16, shuffle=True, num_workers=1, lesion=lesion)
    print ('==>>> total trainning batch number: {}'.format(len(dataloader_train)))
    print('********************load data succeed!********************')
    
    print('********************load VAE model********************')
    # initialize and load the model
    model = BetaVAE(1, 512, loss_type='H', model_type='VAE').cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: "+CKPT_PATH)
    #model = nn.DataParallel(model).cuda()
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3) #default:0.0, L2 = 1e-3
    lr_scheduler_model = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
    print('********************load model succeed!********************')

    print('********************begin training for VAE!********************')
    log_writer = SummaryWriter('/data/tmpexec/tb_log')
    loss_max = float('inf')
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        model.train()  #set model to training mode
        loss_train= []
        with torch.autograd.enable_grad():
            for batch_idx, (img, _, _) in enumerate(dataloader_train):
                optimizer.zero_grad()
                #forward
                var_img = torch.autograd.Variable(img).cuda()
                var_vec = model(var_img)
                # backward and update parameters
                loss_dict = model.loss_function(*var_vec, M_N = 0.005)
                loss_bn = loss_dict['loss']
                loss_bn.backward()
                optimizer.step()
                #show 
                loss_train.append(loss_bn.item())
                sys.stdout.write('\r Train Epoch: {} / Step: {} : loss = {}'.format(epoch+1, batch_idx+1, float('%0.4f'%loss_bn.item()) ))
                sys.stdout.flush()
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Train Eopch: %5d loss = %.4f " % (epoch + 1, np.mean(loss_train) ))

        if loss_max > np.mean(loss_train):
            loss_max = np.mean(loss_train)
            #torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            torch.save(model.state_dict(), CKPT_PATH)
            print('Epoch: {} model has been already save!'.format(epoch + 1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))
        log_writer.add_scalars('IDRID_VAE/mse_kl', {'Train':np.mean(loss_train)}, epoch+1)

    log_writer.close() #shut up the tensorboard

def Query(lesion='MA'):
    print('\r ********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=16, shuffle=False, num_workers=1, lesion=lesion)
    dataloader_test = get_test_dataloader(batch_size=16, shuffle=False, num_workers=1, lesion=lesion)
    #dataloader_test_HE = get_test_dataloader(batch_size=16, shuffle=False, num_workers=1, lesion='HE')
    print('\r ********************load data succeed!********************')

    print('\r ********************load model********************')
    model = BetaVAE(1, 512, loss_type='H', model_type='VAE').cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: "+CKPT_PATH)
    model.eval()#turn to test mode
    print('\r ******************** load model succeed!********************')

    print('\r ********************Build gallery********************')
    db_lbl = torch.FloatTensor().cuda()
    db_vec = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (img, _, lbl) in enumerate(dataloader_train):
            db_lbl = torch.cat((db_lbl, lbl.cuda()), 0)
            var_img = torch.autograd.Variable(img).cuda()
            var_vec = model(var_img)
            db_vec = torch.cat((db_vec, var_vec[4].data), 0)
            sys.stdout.write('\r train set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()
    print('\r ********************Build gallery succeed!********************')

    print('\r ********************Build query********************')
    te_lbl = torch.FloatTensor().cuda()
    te_vec = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (img, _, lbl) in enumerate(dataloader_test):
            te_lbl = torch.cat((te_lbl, lbl.cuda()), 0)
            var_img = torch.autograd.Variable(img).cuda()
            var_vec = model(var_img)
            te_vec = torch.cat((te_vec, var_vec[4].data), 0)
            sys.stdout.write('\r test set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()
    print('\r ********************Build query succeed!********************')

    print('********************Retrieval Performance of {}!********************'.format(lesion))
    sim_mat = cosine_similarity(te_vec.cpu().numpy(), db_vec.cpu().numpy())
    te_lbl = te_lbl.cpu().numpy()
    db_lbl = db_lbl.cpu().numpy()
    mNDCG_dia, nNDCG_num, nNDCG_dis = [], [], [] #diameter, number, distribtuion
    for i in range(sim_mat.shape[0]):
        idxs, vals = zip(*heapq.nlargest(sim_mat.shape[1], enumerate(sim_mat[i,:].tolist()), key=lambda x:x[1]))#
        pd_ndcg_dia, pd_ndcg_num, pd_ndcg_dis = [], [], []
        for j in idxs:
            q_dia, q_num, q_dis = te_lbl[i][0], te_lbl[i][1], te_lbl[i][2]
            r_dia, r_num, r_dis = db_lbl[j][0], db_lbl[j][1], db_lbl[j][2]
            if q_dia < r_dia: pd_ndcg_dia.append(1-q_dia/r_dia)
            else: pd_ndcg_dia.append(1-r_dia/q_dia)
            if q_num < r_num: pd_ndcg_num.append(1-q_num/r_num)
            else: pd_ndcg_num.append(1-r_num/q_num)
            if q_dis < r_dis: pd_ndcg_dis.append(1-q_dis/r_dis)
            else: pd_ndcg_dis.append(1-r_dis/q_dis)

        _, gt_ndcg_dia = zip(*heapq.nlargest(len(pd_ndcg_dia), enumerate(pd_ndcg_dia), key=lambda x:x[1]))
        mNDCG_dia.append(ndcg_score([gt_ndcg_dia], [pd_ndcg_dia]))
        _, gt_ndcg_num = zip(*heapq.nlargest(len(pd_ndcg_num), enumerate(pd_ndcg_num), key=lambda x:x[1]))
        nNDCG_num.append(ndcg_score([gt_ndcg_num], [pd_ndcg_num]))
        _, gt_ndcg_dis = zip(*heapq.nlargest(len(pd_ndcg_dis), enumerate(pd_ndcg_dis), key=lambda x:x[1]))
        nNDCG_dis.append(ndcg_score([gt_ndcg_dis], [pd_ndcg_dis]))
    print("\r NDCG on diameter (NDCG@S) ={:.2f}".format(np.mean(mNDCG_dia)*100))
    print("\r NDCG on number (NDCG@N) ={:.2f}".format(np.mean(nNDCG_num)*100))
    print("\r NDCG on distribution (NDCG@D) ={:.2f}".format(np.mean(nNDCG_dis)*100))
    
def main():
    Train(lesion='MA')#MA training
    Query(lesion='MA')

    #Train(lesion='HE')#HE training
    Query(lesion='HE')

if __name__ == '__main__':
    main()
    #nohup python3 main_idrid_vae.py > logs/main_idrid_vae.log 2>&1 &
