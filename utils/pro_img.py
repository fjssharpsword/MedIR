import torch
import os
import imageio
from PIL import Image
import sys

def genGif():
    images = []
    for i in range(420001):
        if i % 10000 == 0:
            filename = f"/data/pycode/SFSAttention/stylegan/sample/{str(i).zfill(6)}.png"
            if os.path.exists(filename):
                images.append(imageio.imread(filename))
    imageio.mimsave('/data/pycode/SFSAttention/stylegan/imgs/fundus_sample_stylegan.gif', images,fps=1)

def cutImage():
    file_path = '/data/pycode/SFSAttention/stylegan/imgs/cutimgs/IDRiD_81.jpg' 
    save_path = '/data/pycode/SFSAttention/stylegan/imgs/cutimgs/'
    
    def fill_image(image):
        width, height = image.size
        new_image_length = width if width > height else height
        new_image = Image.new(image.mode, (new_image_length, new_image_length), color='white')
        if width > height:
            new_image.paste(image, (0, int((new_image_length - height) / 2)))
        else:
            new_image.paste(image, (int((new_image_length - width) / 2), 0))
        return new_image
    
    def cut_image(image):
        width, height = image.size
        item_width = int(width / 3)
        box_list = []
        for i in range(0, 3):
            for j in range(0, 3):
                box = (j * item_width, i * item_width, (j + 1) * item_width, (i + 1) * item_width)
                box_list.append(box)
        image_list = [image.crop(box) for box in box_list]
        return image_list

    def save_images(image_list,save_path):
        index = 1
        for image in image_list:
            image.save(save_path + str(index) + '.png', 'PNG')
            index += 1

    image = Image.open(file_path)
    #image = fill_image(image)
    image = image.resize((256, 256),Image.ANTIALIAS)
    image_list = cut_image(image)
    save_images(image_list, save_path)

if __name__ == "__main__":
    #genGif()
    cutImage()
    
