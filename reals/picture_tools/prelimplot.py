import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import random 
import sys
import os, subprocess
from pdf2image import convert_from_path
import math
from icecream import ic
import shutil
from PIL import Image, ImageDraw, ImageFont

#This project
from src.utils import filestruct
from src.utils import query_maker

fs = filestruct.fs()

def img_from_pdf(img_dir):
	image_files = []
	lists = os.listdir(img_dir)
	sort_list = sorted(lists)
	for img_file in sort_list:
		print("On file " + img_file)
		image1 = Image.open(img_dir+img_file)
		image_files.append(image1)

	return image_files



def append_images(images, xb_ranges,q2_ranges, xb_counter, direction='horizontal', 
                  bg_color=(255,255,255), aligment='center'):
    
    # Appends images in horizontal/vertical direction.

    # Args:
    #     images: List of PIL images
    #     direction: direction of concatenation, 'horizontal' or 'vertical'
    #     bg_color: Background color (default: white)
    #     aligment: alignment mode if images need padding;
    #        'left', 'right', 'top', 'bottom', or 'center'

    # Returns:
    #     Concatenated image as a new PIL image object.
    
    widths, heights = zip(*(i.size for i in images))

    if direction=='horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)

    if direction=='vertical':
        new_im = Image.new('RGB', (int(new_width+0), int(new_height+images[0].size[1]/2)), color=bg_color)


    offset = 0
    for im_counter,im in enumerate(images):
        ic(im_counter)
        if direction=='horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1])/2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            print("xb counter is {}".format(xb_counter))
            if xb_counter < 0:
            #here we create vertical strip of Q2 values
                offset += im.size[1]

                draw = ImageDraw.Draw(new_im)

                text = str(q2_ranges[im_counter+1])
                textwidth, textheight = images[0].size[0]/5, images[0].size[1]/5

                margin = 10
                #x = images[0].size[0] - textwidth - margin
                #y = images[0].size[1] - textheight - margin
                x = 0.4*int(images[0].size[0])
                ic(im_counter)
                y = 1*(int(images[0].size[1])*(len(q2_ranges)-(im_counter+2)))
                ic(y)
                ic(text)
                fonts_path = os.path.join(fs.base_dir,fs.fonts_dir)
                ic(fonts_path)
                font = ImageFont.truetype(os.path.join(fonts_path, 'agane_bold.ttf'), 150)


                draw.text((x, y), text,(0,0,0),font=font)
            
            else:
                x = 0
                if aligment == 'center':
                    x = int((new_width - im.size[0])/2)
                elif aligment == 'right':
                    x = new_width - im.size[0]
                new_im.paste(im, (x, offset))
            offset += im.size[1]

    if (direction=='vertical') and (xb_counter > -1):
        #This is for the x-axis labels (xB)

        draw = ImageDraw.Draw(new_im)

        text = str(xb_ranges[xb_counter+1])
        textwidth, textheight = images[0].size[0]/5, images[0].size[1]/5

        margin = 10
        #x = images[0].size[0] - textwidth - margin
        #y = images[0].size[1] - textheight - margin
        x = 0.4*int(images[0].size[0])
        y = int(images[0].size[1])*(len(q2_ranges)-1)
        fonts_path = os.path.join(fs.base_dir,fs.fonts_dir)

        font = ImageFont.truetype(os.path.join(fonts_path, 'agane_bold.ttf'), 150)


        draw.text((x, y), text,(0,0,0),font=font)


    return new_im


def chunks(l, n):
	spits = (l[i:i+n] for i in range(0, len(l), n))
	return spits



def stitch_pics(img_dir,xb_ranges,q2_ranges,save_dir="./",fig_name='stiched_pic',t_insert_text="none"):
   
    images = img_from_pdf(img_dir)


    print(len(images))
    #print(images)
    layers = []

    num_ver_slices = len(q2_ranges)-1
    num_hori_slices = len(xb_ranges)-1
    #for i in range(0,int(len(images)/num_ver_slices)):
    for i in range(0,num_hori_slices):
        #print("on step "+str(i))
        layer = list(reversed(images[i*num_ver_slices:i*num_ver_slices+num_ver_slices]))
        #print(layer)
        #list(reversed(array))
        layers.append(layer)

    #print(layers[0])

    horimg = []

    #make vertical axis labels
    imglay1 = append_images(layers[0], xb_ranges,q2_ranges, -1, direction='vertical')
    horimg.append(imglay1)


    for xb_counter,layer in enumerate(layers):
        print("len of layers is {}".format(len(layer)))
        print("counter is {}".format(xb_counter))
        print("On vertical layer {}".format(xb_counter))
        #print(layer)
        imglay = append_images(layer, xb_ranges,q2_ranges,xb_counter, direction='vertical')
        imglay.save("testing1.jpg")
        horimg.append(imglay)

    


    print("Joining images horizontally")
    final = append_images(horimg, xb_ranges,q2_ranges, 0,  direction='horizontal')

    if t_insert_text is not "none":
        draw = ImageDraw.Draw(final)

        text = "t: 0.09-0.15"
        textwidth, textheight = final.size[0]/5, final.size[1]/5

        margin = 10
        #x = images[0].size[0] - textwidth - margin
        #y = images[0].size[1] - textheight - margin
        x = 0.2*int(final.size[0])
        y = int(final.size[1])*0.1
        fonts_path = os.path.join(fs.base_dir,fs.fonts_dir)

        font = ImageFont.truetype(os.path.join(fonts_path, 'agane_bold.ttf'), 250)


        draw.rectangle([.9*x, 0.8*y,2.3*x,2.1*y], fill=(255,255,255), outline=(0,0,0), width=20)
        draw.text((x, y), t_insert_text,font=font,fill=(0,0,0))

    final_name = "{}_{}.jpg".format(fig_name,num_ver_slices)
    print("TRYING TO SAVE AT {}".format(save_dir))
    final.save(save_dir+final_name,optimize=True, quality=100)
    print("saved {}".format(final_name))
