import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from os import listdir
from os.path import isfile, join
import scipy
import openslide
from PIL import Image
#Image.MAX_IMAGE_PIXELS = 933120000
Image.MAX_IMAGE_PIXELS = None
"""
https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data
https://zhuanlan.zhihu.com/p/357125587

Karolinska: Regions are labelled. Valid values are:
0: background (non tissue) or unknown
1: benign tissue (stroma and epithelium combined)
2: cancerous tissue (stroma and epithelium combined)

"""
def transparent_back(img):
    #img = img.convert('RGBA')
    L, H = img.size
    #x_max, x_min, y_max, y_min = 0, 0, 0, 0 
    color_bg = (0,0,0,255)
    color_benign = (1,0,0,255)
    color_tumor = (2,0,0,255)
    for h in range(H):
        for l in range(L):
            dot = (l,h)
            color_dot = img.getpixel(dot)
            if color_dot==color_benign: #benign tissue
                color_dot = color_bg[:-1] + (0,) #alpha=0, transparency 

            elif color_dot==color_tumor: #cancerous tissue
                color_dot = ( 0, 0, 255, 255) #turn to blue  #( 0, 255, 0, 255) #turn to green 
            else:#backgroud and else
                color_dot = color_bg[:-1] + (0,) #alpha=0, transparency
            img.putpixel(dot,color_dot) 
                
    return img

def main():
    fig, axes = plt.subplots(1,5, constrained_layout=True, figsize=(4,2))#

    #isup=5
    #031f5ef5b254fbacd6fbd279ebfe5cc0
    img = Image.open('/data/fjsdata/WSI/PANDA/train_images/0a8c2bda6e00a040372185ccd9a3c4ab.tiff')#.convert('RGBA')
    img = img.resize((1024, 1024),Image.ANTIALIAS)
    mask = Image.open('/data/fjsdata/WSI/PANDA/train_label_masks/0a8c2bda6e00a040372185ccd9a3c4ab_mask.tiff')#.convert('RGBA')
    mask = mask.resize((1024, 1024),Image.ANTIALIAS)
    #df = pd.DataFrame(np.array(mask.getdata()), columns=['R','G','B','A'])
    #print(df['R'].value_counts())
    #print(df['G'].value_counts())
    #print(df['B'].value_counts())
    #print(df['A'].value_counts())
    bbox = mask.getbbox() #get non-zero pixel regions
    mask = transparent_back(mask.convert('RGBA'))
    mask = mask.crop(bbox)
    img = img.crop(bbox)
    img_mask = Image.alpha_composite(img.convert('RGBA'), mask)
    #img_mask.save('/data/pycode/MedIR/WSI/imgs/panda_isup_test.png')
    axes[0].imshow(img_mask, aspect="auto")
    axes[0].axis('off')
    axes[0].set_title('Grade 5')
    axes[0].vlines(img_mask.size[0], 0, img_mask.size[1], linestyles='dashed', colors='red')

    #isup=4
    #001c62abd11fa4b57bf7a6c603a11bb9,020d0243094c8561544f683b4c64859d
    img = Image.open('/data/fjsdata/WSI/PANDA/train_images/04e1a953bceb9b1a0a5bf8840614aa04.tiff')#.convert('RGBA')
    img = img.resize((1024, 1024),Image.ANTIALIAS)
    mask = Image.open('/data/fjsdata/WSI/PANDA/train_label_masks/04e1a953bceb9b1a0a5bf8840614aa04_mask.tiff')#.convert('RGBA')
    mask = mask.resize((1024, 1024),Image.ANTIALIAS)
    bbox = mask.getbbox()
    mask = transparent_back(mask.convert('RGBA'))
    mask = mask.crop(bbox)
    img = img.crop(bbox)
    img_mask = Image.alpha_composite(img.convert('RGBA'), mask)
    axes[1].imshow(img_mask, aspect="auto")
    axes[1].axis('off')
    axes[1].set_title('Grade 4')
    axes[1].vlines(img_mask.size[0], 0, img_mask.size[1], linestyles='dashed', colors='red')

    #isup=3
    #010f9df31ea44191c106d8226eaf46fb
    img = Image.open('/data/fjsdata/WSI/PANDA/train_images/02533ec710c56377e6df314abc2d6589.tiff')#.convert('RGBA')
    img = img.resize((1024, 1024),Image.ANTIALIAS)
    mask = Image.open('/data/fjsdata/WSI/PANDA/train_label_masks/02533ec710c56377e6df314abc2d6589_mask.tiff')#.convert('RGBA')
    mask = mask.resize((1024, 1024),Image.ANTIALIAS)
    bbox = mask.getbbox()
    mask = transparent_back(mask.convert('RGBA'))
    mask = mask.crop(bbox)
    img = img.crop(bbox)
    img_mask = Image.alpha_composite(img.convert('RGBA'), mask)
    axes[2].imshow(img_mask, aspect="auto")
    axes[2].axis('off')
    axes[2].set_title('Grade 3')
    axes[2].vlines(img_mask.size[0], 0, img_mask.size[1], linestyles='dashed', colors='red')

    #isup=2
    img = Image.open('/data/fjsdata/WSI/PANDA/train_images/00d7ec94436e3a1416a3b302914957d3.tiff')#.convert('RGBA')
    img = img.resize((1024, 1024),Image.ANTIALIAS)
    mask = Image.open('/data/fjsdata/WSI/PANDA/train_label_masks/00d7ec94436e3a1416a3b302914957d3_mask.tiff')#.convert('RGBA')
    mask = mask.resize((1024, 1024),Image.ANTIALIAS)
    bbox = mask.getbbox()
    mask = transparent_back(mask.convert('RGBA'))
    mask = mask.crop(bbox)
    img = img.crop(bbox)
    img_mask = Image.alpha_composite(img.convert('RGBA'), mask)
    axes[3].imshow(img_mask, aspect="auto")
    axes[3].axis('off')
    axes[3].set_title('Grade 2')
    axes[3].vlines(img_mask.size[0], 0, img_mask.size[1], linestyles='dashed', colors='red')

    #isup=1
    img = Image.open('/data/fjsdata/WSI/PANDA/train_images/003046e27c8ead3e3db155780dc5498e.tiff')#.convert('RGBA')
    img = img.resize((1024, 1024),Image.ANTIALIAS)
    mask = Image.open('/data/fjsdata/WSI/PANDA/train_label_masks/003046e27c8ead3e3db155780dc5498e_mask.tiff')#.convert('RGBA')
    mask = mask.resize((1024, 1024),Image.ANTIALIAS)
    bbox = mask.getbbox()
    mask = transparent_back(mask.convert('RGBA'))
    mask = mask.crop(bbox)
    img = img.crop(bbox)
    img_mask = Image.alpha_composite(img.convert('RGBA'), mask)
    axes[4].imshow(img_mask, aspect="auto")
    axes[4].axis('off')
    axes[4].set_title('Grade 1')


    fig.savefig('/data/pycode/MedIR/WSI/imgs/panda_isup_seg.png', dpi=300, bbox_inches='tight', pad_inches=0)

def slidetest():
    slide = openslide.OpenSlide('/data/pycode/MedIR/WSI/jpg/0005f7aaab2800f6170c399693a96917.tiff')
    mask = openslide.OpenSlide('/data/pycode/MedIR/WSI/jpg/0005f7aaab2800f6170c399693a96917_mask.tiff')
    print(slide.level_count)
    print(slide.dimensions)
    print(mask.dimensions)
    slide_thumbnail = slide.get_thumbnail((1528,3432))
    slide_thumbnail = np.array(slide_thumbnail)
    slide_thumbnail=np.array(slide.read_region((0,0), 1, (6912, 7360)))
    plt.imsave('/data/pycode/MedIR/WSI/jpg/orig.jpg', slide_thumbnail)
    mask_thumbnail = mask.get_thumbnail((1528,3432))
    mask_thumbnail = np.array(mask_thumbnail)
    mask_thumbnail=np.array(mask.read_region((0,0), 1, (6912, 7360)))
    plt.imsave('/data/pycode/MedIR/WSI/jpg/mask.jpg', mask_thumbnail)

def test():
    img = Image.open('/data/fjsdata/WSI/PANDA/train_images/04e1a953bceb9b1a0a5bf8840614aa04.tiff')#.convert('RGBA')
    img = img.resize((512, 512),Image.ANTIALIAS)
    mask = Image.open('/data/fjsdata/WSI/PANDA/train_label_masks/04e1a953bceb9b1a0a5bf8840614aa04_mask.tiff')#.convert('RGBA')
    mask = mask.resize((512, 512),Image.ANTIALIAS)

    bbox = mask.getbbox()
    print(bbox)
    mask = mask.convert('RGBA')
    mask = transparent_back(mask)

    mask = mask.crop(bbox)
    img = img.crop(bbox)
    img = img.convert('RGBA')

    img_mask = Image.alpha_composite(img.convert('RGBA'), mask.convert('RGBA'))
    
    img_mask.save('/data/pycode/MedIR/WSI/imgs/panda_isup_test.png')

if __name__ == '__main__':
    main()
    #test()
    
    
    
    