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


def main():
    slide = openslide.OpenSlide('/data/pycode/MedIR/WSI/jpg/0005f7aaab2800f6170c399693a96917.tiff')
    mask = openslide.OpenSlide('/data/pycode/MedIR/WSI/jpg/0005f7aaab2800f6170c399693a96917_mask.tiff')
    #print(slide.level_count)
    #print(slide.dimensions)
    #print(mask.dimensions)
    #slide_thumbnail = slide.get_thumbnail((1528,3432))
    #slide_thumbnail = np.array(slide_thumbnail)
    slide_thumbnail=np.array(slide.read_region((0,0), 1, (6912, 7360)))
    plt.imsave('/data/pycode/MedIR/WSI/jpg/orig.jpg', slide_thumbnail)
    #mask_thumbnail = mask.get_thumbnail((1528,3432))
    #mask_thumbnail = np.array(mask_thumbnail)
    mask_thumbnail=np.array(mask.read_region((0,0), 1, (6912, 7360)))
    plt.imsave('/data/pycode/MedIR/WSI/jpg/mask.jpg', mask_thumbnail)


if __name__ == '__main__':
    main()