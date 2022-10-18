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
from nets.vae import BetaVAE, MSE_Loss
from nets.unet_2d import UNet, KL_Loss, Dice_Loss

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "5,6"
UNet_CKPT_PATH = '/data/pycode/MedIR/fundus/ckpts/idrid_unet.pkl'
VAE_CKPT_PATH = '/data/pycode/MedIR/fundus/ckpts/idrid_vae.pkl'
MAX_EPOCHS = 100

def Train(lesion='MA'):
    print('********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=16, shuffle=True, num_workers=1, lesion=lesion)
    print ('==>>> total trainning batch number: {}'.format(len(dataloader_train)))
    print('********************load data succeed!********************')
   
    print('********************load unet model********************')
    unet_model = UNet(n_channels=3, n_classes=1, latent_dim=512).cuda()
    if os.path.exists(UNet_CKPT_PATH):
        checkpoint = torch.load(UNet_CKPT_PATH)
        unet_model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: "+UNet_CKPT_PATH)
    unet_optimizer = optim.Adam(unet_model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    unet_lr_schedule = lr_scheduler.StepLR(unet_optimizer , step_size = 10, gamma = 1)
    kl_criterion = KL_Loss()

    print('********************load VAE model********************')
    vae_model = BetaVAE(3, 512, loss_type='H', model_type='VAE').cuda() # initialize and load the model
    if os.path.exists(VAE_CKPT_PATH):
        checkpoint = torch.load(VAE_CKPT_PATH)
        vae_model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: "+VAE_CKPT_PATH)
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3) #default:0.0, L2 = 1e-3
    vae_lr_scheduler = lr_scheduler.StepLR(vae_optimizer, step_size=10, gamma=1)
    print('********************load model succeed!********************')

    print('********************begin training for VAE-UNet!********************')
    log_writer = SummaryWriter('/data/tmpexec/tb_log')
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    loss_min_mse, loss_min_dice = float('inf'), float('inf')
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        unet_model.train()  #set model to training mode
        vae_model.train()
        loss_train = []
        mse_loss_list, dice_loss_list, kl_loss_list  = [], [], []
        with torch.autograd.enable_grad():
            for batch_idx, (img, msk, _) in enumerate(dataloader_train):
                unet_optimizer.zero_grad()
                vae_optimizer.zero_grad()
                #forward
                var_img = torch.autograd.Variable(img).cuda()
                var_msk = torch.autograd.Variable(msk).cuda()
                
                out_unet = unet_model(var_img)
                out_vae = vae_model(var_img)
                # backward and update parameters
                dice_loss = Dice_Loss(var_msk, out_unet[0])
                mse_loss = MSE_Loss(out_vae)
                kl_loss = 0.1 * kl_criterion(out_unet[3], out_vae[4])
                loss_bn =  dice_loss + mse_loss + kl_loss
                loss_bn.backward()
                unet_optimizer.step()
                vae_optimizer.step()
                #show 
                loss_train.append(loss_bn.item())
                mse_loss_list.append(mse_loss.item())
                dice_loss_list.append(dice_loss.item())
                kl_loss_list.append(kl_loss.item())
                sys.stdout.write('\r Train Epoch: {} / Step: {} : loss = {}'.format(epoch+1, batch_idx+1, float('%0.4f'%loss_bn.item()) ))
                sys.stdout.flush()
        unet_lr_schedule.step()  #about lr and gamma
        vae_lr_scheduler.step()  #about lr and gamma
        print("\r Train Eopch: %5d Train loss = %.4f, Dice loss=%.4f, MSE loss=%.4f, KL loss =%.4f" % (epoch + 1, np.mean(loss_train), np.mean(dice_loss_list), np.mean(mse_loss_list), np.mean(kl_loss_list) ))

        if loss_min_mse > np.mean(mse_loss_list):
            loss_min_mse = np.mean(mse_loss_list)
            torch.save(vae_model.state_dict(), VAE_CKPT_PATH)
            print('Epoch: {} VAE model has been already save!'.format(epoch + 1))
        if loss_min_dice> np.mean(dice_loss_list):
            loss_min_dice = np.mean(dice_loss_list)
            torch.save(unet_model.state_dict(), UNet_CKPT_PATH)
            print('Epoch: {} UNet model has been already save!'.format(epoch + 1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))
        log_writer.add_scalars('IDRID_VAE_UNet/dice_mse_kl', {'All':np.mean(loss_train), 'Dice': np.mean(dice_loss_list), 'MSE': np.mean(mse_loss_list), 'KL': np.mean(kl_loss_list)}, epoch+1)

    log_writer.close() #shut up the tensorboard

def Query(lesion='MA'):
    print('\r ********************load data********************')
    dataloader_train = get_train_dataloader(batch_size=16, shuffle=False, num_workers=1, lesion=lesion)
    dataloader_test = get_test_dataloader(batch_size=16, shuffle=False, num_workers=1, lesion=lesion)
    print('\r ********************load data succeed!********************')

    print('\r ********************load model********************')
    vae_model = BetaVAE(3, 512, loss_type='H', model_type='VAE').cuda()
    if os.path.exists(VAE_CKPT_PATH):
        checkpoint = torch.load(VAE_CKPT_PATH)
        vae_model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: "+VAE_CKPT_PATH)
    vae_model.eval()#turn to test mode

    unet_model = UNet(n_channels=3, n_classes=1, latent_dim=512).cuda()
    if os.path.exists(UNet_CKPT_PATH):
        checkpoint = torch.load(UNet_CKPT_PATH)
        unet_model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: "+UNet_CKPT_PATH)
    unet_model.eval()#turn to test mode
    print('\r ******************** load model succeed!********************')

    print('\r ********************Build gallery********************')
    db_lbl = torch.FloatTensor().cuda()
    db_vec = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (img, _, lbl) in enumerate(dataloader_train):
            db_lbl = torch.cat((db_lbl, lbl.cuda()), 0)
            var_img = torch.autograd.Variable(img).cuda()
            var_vec = vae_model(var_img)
            db_vec = torch.cat((db_vec, var_vec[4].data), 0)
            sys.stdout.write('\r train set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()
    print('\r ********************Build gallery succeed!********************')

    print('\r ********************Build query********************')
    te_lbl = torch.FloatTensor().cuda()
    te_vec = torch.FloatTensor().cuda()
    dice_coe = []
    with torch.autograd.no_grad():
        for batch_idx, (img, msk, lbl) in enumerate(dataloader_test):
            te_lbl = torch.cat((te_lbl, lbl.cuda()), 0)
            var_img = torch.autograd.Variable(img).cuda()
            var_msk = torch.autograd.Variable(msk).cuda()
            var_vec = vae_model(var_img)
            te_vec = torch.cat((te_vec, var_vec[4].data), 0)
            out_unet = unet_model(var_img)
            dice_coe.append(Dice_Loss(var_msk, out_unet[0]).item())
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
    print("\r Dice coefficient = %.2f" % (100- np.mean(dice_coe)*100))
    
def main():
    Train(lesion='MA')#MA training
    Query(lesion='MA')

    #Train(lesion='HE')#HE training
    #Query(lesion='HE')

if __name__ == '__main__':
    main()
    #nohup python3 main_idrid_vae_unet.py > logs/main_idrid_vae_unet.log 2>&1 &
