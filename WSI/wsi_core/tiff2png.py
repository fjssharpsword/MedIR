import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from os import listdir
from os.path import isfile, join
import scipy
import openslide
from PIL import Image

def transparent_back(img):
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
                color_1 = ( 0, 0, 255, 255) #turn to blue  and transparency 
                #color_1 = ( 0 , 255, 0, 255) #turn to green  and transparency 
                img.putpixel(dot,color_1)
    return img

def main():
    #slide = openslide.OpenSlide('/data/pycode/MedIR/WSI/jpg/0005f7aaab2800f6170c399693a96917.tiff')
    #mask = openslide.OpenSlide('/data/pycode/MedIR/WSI/jpg/0005f7aaab2800f6170c399693a96917_mask.tiff')
    #print(slide.level_count)
    #print(slide.dimensions)
    #print(mask.dimensions)
    #slide_thumbnail = slide.get_thumbnail((1528,3432))
    #slide_thumbnail = np.array(slide_thumbnail)
    #slide_thumbnail=np.array(slide.read_region((0,0), 1, (6912, 7360)))
    #plt.imsave('/data/pycode/MedIR/WSI/jpg/orig.jpg', slide_thumbnail)
    #mask_thumbnail = mask.get_thumbnail((1528,3432))
    #mask_thumbnail = np.array(mask_thumbnail)
    #mask_thumbnail=np.array(mask.read_region((0,0), 1, (6912, 7360)))
    #plt.imsave('/data/pycode/MedIR/WSI/jpg/mask.jpg', mask_thumbnail)

    fig, axes = plt.subplots(1,5, constrained_layout=True, figsize=(25,5))

    #isup=5
    img = Image.open('/data/fjsdata/WSI/PANDAS/train_images/031f5ef5b254fbacd6fbd279ebfe5cc0.tiff').convert('RGBA')
    mask = Image.open('/data/fjsdata/WSI/PANDAS/train_label_masks/031f5ef5b254fbacd6fbd279ebfe5cc0_mask.tiff').convert('RGBA')
    mask = transparent_back(mask)
    img_mask = Image.alpha_composite(img, mask)
    axes[0].imshow(img_mask, aspect="auto")
    axes[0].axis('off')
    axes[0].set_title('ISUP grade 5')

    #isup=4
    img = Image.open('/data/fjsdata/WSI/PANDAS/train_images/001c62abd11fa4b57bf7a6c603a11bb9.tiff').convert('RGBA')
    mask = Image.open('/data/fjsdata/WSI/PANDAS/train_label_masks/001c62abd11fa4b57bf7a6c603a11bb9_mask.tiff').convert('RGBA')
    mask = transparent_back(mask)
    img_mask = Image.alpha_composite(img, mask)
    axes[0].imshow(img_mask, aspect="auto")
    axes[0].axis('off')
    axes[0].set_title('ISUP grade 4')

    #isup=3
    img = Image.open('/data/fjsdata/WSI/PANDAS/train_images/010f9df31ea44191c106d8226eaf46fb.tiff').convert('RGBA')
    mask = Image.open('/data/fjsdata/WSI/PANDAS/train_label_masks/010f9df31ea44191c106d8226eaf46fb_mask.tiff').convert('RGBA')
    mask = transparent_back(mask)
    img_mask = Image.alpha_composite(img, mask)
    axes[0].imshow(img_mask, aspect="auto")
    axes[0].axis('off')
    axes[0].set_title('ISUP grade 3')

    #isup=2
    img = Image.open('/data/fjsdata/WSI/PANDAS/train_images/00d7ec94436e3a1416a3b302914957d3.tiff').convert('RGBA')
    mask = Image.open('/data/fjsdata/WSI/PANDAS/train_label_masks/00d7ec94436e3a1416a3b302914957d3_mask.tiff').convert('RGBA')
    mask = transparent_back(mask)
    img_mask = Image.alpha_composite(img, mask)
    axes[0].imshow(img_mask, aspect="auto")
    axes[0].axis('off')
    axes[0].set_title('ISUP grade 2')

    #isup=1
    img = Image.open('/data/fjsdata/WSI/PANDAS/train_images/003046e27c8ead3e3db155780dc5498e.tiff').convert('RGBA')
    mask = Image.open('/data/fjsdata/WSI/PANDAS/train_label_masks/003046e27c8ead3e3db155780dc5498e_mask.tiff').convert('RGBA')
    mask = transparent_back(mask)
    img_mask = Image.alpha_composite(img, mask)
    axes[0].imshow(img_mask, aspect="auto")
    axes[0].axis('off')
    axes[0].set_title('ISUP grade 1')


    fig.savefig('/data/pycode/MedIR/WSI/imgs/pandas_isup_seg.png', dpi=300, bbox_inches='tight')



if __name__ == '__main__':
    main()