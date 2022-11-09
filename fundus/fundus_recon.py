# encoding: utf-8
"""
Training implementation for DDR grading dataset.
Author: Jason.Fang
Update time: 17/10/2022
"""
import sys
import os
import cv2
import time
import heapq
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
#self-defined
from dsts.idrid_seg import get_train_dataloader, get_test_dataloader
from nets.vae import BetaVAE
from nets.unet_2d import UNet

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "5,6"
CKPT_PATH = '/data/pycode/MedIR/fundus/ckpts/idrid_vae.pkl'

def fundus_recon(lesion='MA'):
    print('\r ********************load model********************')
    model = BetaVAE(1, 512, loss_type='H', model_type='VAE').cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: "+CKPT_PATH)
    model.eval()#turn to test mode
    print('\r ******************** load model succeed!********************')

    img_path = '/data/fjsdata/fundus/IDRID/ASegmentation/Images/TrainingSet/IDRiD_18.jpg'
    #in_img = Image.open(img_path).convert('L')
    transform_seq = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
            ])
 
    in_img = transform_seq(Image.open(img_path).convert('L'))
    var_img = torch.autograd.Variable(torch.unsqueeze(in_img,0)).cuda()
    var_out = model.generate(var_img)
    var_out = var_out.cpu().detach().numpy()
    out_img = var_out.squeeze(0).transpose(1,2,0)
    #out_img = np.uint8(out_img * 255.0)
    #out_img = cv2.cvtColor(np.uint8(out_img * 255.0), cv2.COLOR_BGR2RGB)
    plt.imshow(out_img)
    plt.axis('off')
    plt.savefig('/data/pycode/MedIR/fundus/imgs/IDRiD_18_recon.png')

    
def main():
    fundus_recon(lesion='MA')

if __name__ == '__main__':
    main()
    #nohup python3 main_idrid_vae.py > logs/main_idrid_vae.log 2>&1 &
