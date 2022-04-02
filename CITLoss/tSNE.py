#https://blog.csdn.net/fjssharpsword/article/details/104982459
#visualize : t-SNE
import os
import sys
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib import cm
import random
import heapq
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics import ndcg_score
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, confusion_matrix, accuracy_score
#self-defined
import torch
from resnet import resnet50
from vit import ViT
from vincxr_cls import get_box_dataloader_VIN
#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
CLASS_NAMES = ['No finding', 'Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
                'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
CKPT_PATH = '/data/pycode/MedIR/CITLoss/ckpts/vincxr_vit.pkl'
MAX_EPOCHS = 50
BATCH_SIZE = 16*8

def scatter(X, y):
    #X,y:numpy-array
    labels = len(list(set(y.tolist())))#get number of classes
    #palette = np.array(sns.color_palette("hls", labels))# choose a color palette with seaborn.
    colors = cm.rainbow(np.linspace(0,1,labels))  #colors = ['c','y','m','b','g','r']
    #marker = ['o','x','+','*','s']
    plt.figure(figsize=(8,8))#create a plot
    for i in range(labels):
        #plt.scatter(X[y == i,0], X[y == i,1], c=colors[i], marker=marker[i], label=str(i))
        plt.scatter(X[y == i,0], X[y == i,1], c=colors[i], label=CLASS_NAMES[i])
    plt.axis('off')
    plt.legend(loc='lower left')
    plt.savefig('/data/pycode/MedIR/CITLoss/imgs/vis_cluster_vincxr.png', dpi=300, bbox_inches='tight')
    #plt.show()

def Vis_tSNE_VinCXR():
    print('********************load data********************')
    train_loader = get_box_dataloader_VIN(batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_loader = get_box_dataloader_VIN(batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print ('==>>> total test batch number: {}'.format(len(test_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    #model = resnet50(pretrained=False, num_classes=len(CLASS_NAMES)*20).cuda()
    model = ViT(image_size = 224, patch_size = 32, num_classes = len(CLASS_NAMES)*20, dim = 1024, depth = 6,heads = 16, mlp_dim = 2048).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: "+CKPT_PATH)
    model.eval()#turn to test mode
    print('******************** load model succeed!********************')

    print('********************tSNE!********************')
    """
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
    """
    te_label = torch.FloatTensor().cuda()
    te_feat = torch.FloatTensor().cuda()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(test_loader):
            te_label = torch.cat((te_label, label.cuda()), 0)
            var_image = torch.autograd.Variable(image).cuda()
            var_label = torch.autograd.Variable(label).cuda()
            var_feat = model(var_image)
            te_feat = torch.cat((te_feat, var_feat.data), 0)
            sys.stdout.write('\r test set process: = {}'.format(batch_idx + 1))
            sys.stdout.flush()

    print('********************tSNE cluster visulazation********************')
    #tr_feat = tr_feat.cpu().numpy()
    #tr_label = tr_label.cpu().numpy()
    te_feat = te_feat.cpu().numpy()
    te_label = te_label.cpu().numpy()
    
    #training t-sne 
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    tsne_te_feat = tsne.fit_transform(te_feat)
    print("Org data dimension is {}.Embedded data dimension is {}".format(te_feat.shape[-1], tsne_te_feat.shape[-1]))  
    #visualize
    scatter(tsne_te_feat, te_label) 


def main():
    Vis_tSNE_VinCXR() #for test

if __name__ == '__main__':
    main()