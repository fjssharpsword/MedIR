
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.measure import label as skilabel
from skimage.measure import regionprops
from scipy.ndimage import zoom
import seaborn as sns
#import skimage
#print(skimage.__version__) #0.19.3
#https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_regionprops.py

def transparent_back(img, cls='he'):
    #img = img.convert('RGBA')
    L, H = img.size
    color_0 = img.getpixel((0,0)) #alpha channel: 0~255
    for h in range(H):
        for l in range(L):
            dot = (l,h)
            color_1 = img.getpixel(dot)
            if color_1 == color_0:
                color_1 = color_1[:-1] + (0,)
                img.putpixel(dot,color_1)
            else: 
                if cls=='ma':
                    color_1 = ( 0, 0, 255, 255) #turn to blue  and transparency 
                    img.putpixel(dot,color_1)
                elif cls=='he': 
                    color_1 = ( 0 , 255, 0, 255) #turn to green  and transparency 
                    img.putpixel(dot,color_1)
                else: #od
                    color_1 = ( 0 , 255, 255, 255) #turn to white  and transparency 
                    img.putpixel(dot,color_1)
    return img

def show_lesion():
    img = '/data/fjsdata/fundus/IDRID/ASegmentation/Images/TrainingSet/IDRiD_18.jpg'
    img = Image.open(img).convert('RGBA')
    MA = '/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/Microaneurysms/IDRiD_18_MA.tif'
    MA = Image.open(MA).convert('RGBA')
    trans_MA = transparent_back(MA, 'ma')
    HE = '/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/Haemorrhages/IDRiD_18_HE.tif'
    HE = Image.open(HE).convert('RGBA')
    trans_HE = transparent_back(HE, 'he')
    OD = '/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/OpticDisc/IDRiD_18_OD.tif'
    #OD = Image.open(OD).convert('RGBA')
    #trans_OD = transparent_back(OD, 'od')
    #get centroids of OD
    cv_od = cv2.imread(OD, cv2.IMREAD_GRAYSCALE)
    lbl_od = skilabel(cv_od, 2) #connectivity=Eight connected
    props = regionprops(lbl_od)
    cen_od = props[1].centroid #tuple (row, col) - (w, h)
    
    #plot  
    fig, axes = plt.subplots(1, 3, constrained_layout=True) 
    axes[0].imshow(img)
    axes[0].axis('off')
    #add cordinates
    draw =ImageDraw.Draw(img)
    draw.line([(cen_od[1], 0), (cen_od[1], img.size[1])], fill ='white', width = 20)
    draw.line([(0, cen_od[0]), (img.size[0], cen_od[0])], fill ='white', width = 20)
    #merge
    img_ma = Image.alpha_composite(img, trans_MA)
    axes[1].imshow(img_ma)
    axes[1].axis('off')
    img_he = Image.alpha_composite(img, trans_HE)
    axes[2].imshow(img_he)
    axes[2].axis('off')
    fig.savefig('/data/pycode/MedIR/fundus/imgs/IDRiD_18_IR.png', dpi=300, bbox_inches='tight')
    

def zoom_lesion():
    img = '/data/fjsdata/fundus/IDRID/ASegmentation/Images/TrainingSet/IDRiD_18.jpg'
    img = Image.open(img)

    MA = '/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/Microaneurysms/IDRiD_18_MA.tif'
    #MA = '/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/HardExudates/IDRiD_18_EX.tif'
    cv_ma = cv2.imread(MA, cv2.IMREAD_GRAYSCALE)
    lbl_ma = skilabel(cv_ma, 2) #connectivity=Eight connected
    props_ma = regionprops(lbl_ma)

    HE = '/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/Haemorrhages/IDRiD_18_HE.tif'
    cv_he = cv2.imread(HE, cv2.IMREAD_GRAYSCALE)
    lbl_he = skilabel(cv_he, 2) #connectivity=Eight connected
    props_he = regionprops(lbl_he)
    #obtain lesion rectangle
    area, idx = 0.0, 0
    for i in range(1, len(props_ma)):
        if area < props_ma[i].area:
            area = props_ma[i].area
            idx = i
    p_ma =  props_ma[idx]
    area, idx = 0.0, 0
    for i in range(1, len(props_he)):
        if area < props_he[i].area:
            area = props_he[i].area
            idx = i
    p_he =  props_he[idx]
    p_ma = props_he[3]
    p_he = props_he[4]

    #plot  
    ma_bbox = p_ma.bbox#(min_row, min_col, max_row, max_col).
    he_bbox = p_he.bbox  

    fig, axes = plt.subplots(1, 3, constrained_layout=True) 

    axes[0].imshow(img)
    rect = mpatches.Rectangle((ma_bbox[1]-20, ma_bbox[0]-20), ma_bbox[3] - ma_bbox[1] + 20, ma_bbox[2] - ma_bbox[0] + 20, fill=False, edgecolor='blue', linewidth=0.2)
    axes[0].add_patch(rect)
    rect = mpatches.Rectangle((he_bbox[1]-20, he_bbox[0]-20), he_bbox[3] - he_bbox[1] + 20, he_bbox[2] - he_bbox[0] + 20, fill=False, edgecolor='green', linewidth=0.2)
    axes[0].add_patch(rect)
    axes[0].axis('off')


    lesion_img = Image.new('RGB', (img.size[0], img.size[1]), color=0) #transparency
    ma_img = img.crop((ma_bbox[1], ma_bbox[0], ma_bbox[3], ma_bbox[2]))
    ma_img = ma_img.resize((img.size[0]//2, img.size[1]//2), resample=0)
    lesion_img.paste(ma_img, (img.size[0]//4, img.size[1]//4))
    axes[1].imshow(lesion_img)
    rect = mpatches.Rectangle((img.size[0]//4, img.size[1]//4), img.size[0]//2, img.size[1]//2, fill=False, edgecolor='blue', linewidth=1.0)
    axes[1].add_patch(rect)
    axes[1].axis('off')

    lesion_img = Image.new('RGB', (img.size[0], img.size[1]), color=0) #transparency
    he_img = img.crop((he_bbox[1], he_bbox[0], he_bbox[3], he_bbox[2]))
    he_img = he_img.resize((img.size[0]//2, img.size[1]//2),resample=0)
    lesion_img.paste(he_img, (img.size[0]//4, img.size[1]//4))
    axes[2].imshow(lesion_img)
    rect = mpatches.Rectangle((img.size[0]//4, img.size[1]//4), img.size[0]//2, img.size[1]//2, fill=False, edgecolor='green', linewidth=1.0)
    axes[2].add_patch(rect)
    axes[2].axis('off')

    fig.savefig('/data/pycode/MedIR/fundus/imgs/IDRiD_18_SR.png', dpi=300, bbox_inches='tight')

def lesion_variance():
    img = '/data/fjsdata/fundus/IDRID/ASegmentation/Images/TrainingSet/IDRiD_18.jpg'
    img = Image.open(img)

    MA = '/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/Microaneurysms/IDRiD_18_MA.tif'
    #MA = '/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/HardExudates/IDRiD_18_EX.tif'
    cv_ma = cv2.imread(MA, cv2.IMREAD_GRAYSCALE)
    lbl_ma = skilabel(cv_ma, 2) #connectivity=Eight connected
    props_ma = regionprops(lbl_ma)

    HE = '/data/fjsdata/fundus/IDRID/ASegmentation/Masks/TrainingSet/Haemorrhages/IDRiD_18_HE.tif'
    cv_he = cv2.imread(HE, cv2.IMREAD_GRAYSCALE)
    lbl_he = skilabel(cv_he, 2) #connectivity=Eight connected
    props_he = regionprops(lbl_he)
    #obtain lesion rectangle
    area, idx = 0.0, 0
    for i in range(1, len(props_ma)):
        if area < props_ma[i].area:
            area = props_ma[i].area
            idx = i
    p_ma =  props_ma[idx]
    area, idx = 0.0, 0
    for i in range(1, len(props_he)):
        if area < props_he[i].area:
            area = props_he[i].area
            idx = i
    p_he =  props_he[idx]
    p_ma = props_he[3]
    p_he = props_he[4]

    #plot  
    ma_bbox = p_ma.bbox#(min_row, min_col, max_row, max_col).
    he_bbox = p_he.bbox  

    fig, axes = plt.subplots(2, 2, constrained_layout=True) 

    axes[0,0].imshow(img)
    rect = mpatches.Rectangle((ma_bbox[1]-20, ma_bbox[0]-20), ma_bbox[3] - ma_bbox[1] + 20, ma_bbox[2] - ma_bbox[0] + 20, fill=False, edgecolor='blue', linewidth=0.2)
    axes[0,0].add_patch(rect)
    rect = mpatches.Rectangle((he_bbox[1]-20, he_bbox[0]-20), he_bbox[3] - he_bbox[1] + 20, he_bbox[2] - he_bbox[0] + 20, fill=False, edgecolor='green', linewidth=0.2)
    axes[0,0].add_patch(rect)
    axes[0,0].axis('off')

    
    ma_img = img.crop((ma_bbox[1], ma_bbox[0], ma_bbox[3], ma_bbox[2]))
    ma_img = np.array(ma_img).flatten()
    info_s = r'$\ MA=%.2f$' %(np.std(ma_img))
    sns.distplot(ma_img, kde=True, ax=axes[0,1], hist_kws={'color':'green'}, kde_kws={'color':'green'}, label=info_s)
    he_img = img.crop((he_bbox[1], he_bbox[0], he_bbox[3], he_bbox[2]))
    he_img = np.array(he_img).flatten()
    #print(np.var(np.concatenate((ma_img,he_img))))
    info_s = r'$\ HE=%.2f$' %(np.std(he_img))
    sns.distplot(he_img, kde=True, ax=axes[0,1], hist_kws={'color':'blue'}, kde_kws={'color':'blue'}, label=info_s)
    axes[0,1].grid(b=True, ls=':')
    axes[0,1].legend()
    axes[0,1].set_title('Standard deviation')


    lesion_img = Image.new('RGB', (img.size[0], img.size[1]), color=0) #transparency
    ma_img = img.crop((ma_bbox[1], ma_bbox[0], ma_bbox[3], ma_bbox[2]))
    ma_img = ma_img.resize((img.size[0]//2, img.size[1]//2), resample=0)
    lesion_img.paste(ma_img, (img.size[0]//4, img.size[1]//4))
    axes[1,0].imshow(lesion_img)
    rect = mpatches.Rectangle((img.size[0]//4, img.size[1]//4), img.size[0]//2, img.size[1]//2, fill=False, edgecolor='blue', linewidth=1.0)
    axes[1,0].add_patch(rect)
    axes[1,0].axis('off')

    lesion_img = Image.new('RGB', (img.size[0], img.size[1]), color=0) #transparency
    he_img = img.crop((he_bbox[1], he_bbox[0], he_bbox[3], he_bbox[2]))
    he_img = he_img.resize((img.size[0]//2, img.size[1]//2),resample=0)
    lesion_img.paste(he_img, (img.size[0]//4, img.size[1]//4))
    axes[1,1].imshow(lesion_img)
    rect = mpatches.Rectangle((img.size[0]//4, img.size[1]//4), img.size[0]//2, img.size[1]//2, fill=False, edgecolor='green', linewidth=1.0)
    axes[1,1].add_patch(rect)
    axes[1,1].axis('off')

    fig.savefig('/data/pycode/MedIR/fundus/imgs/IDRiD_18_SR_variance.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    #show_lesion()
    #zoom_lesion()
    lesion_variance()