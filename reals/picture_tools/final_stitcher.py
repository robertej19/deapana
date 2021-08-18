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

   



#xb_ranges = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.85,1]
#q2_ranges = [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,7.0,8.0,9.0,12.0]
#t_ranges = [0.09,0.15,0.2,0.3,0.4,0.6,1,1.5,2,3,4.5,6]

xb_ranges = [0.05,0.125,0.175,0.225,0.275,0.34,0.43,0.53,0.63,0.79]
q2_ranges = [1.25,1.75,2.25,2.75,3.25,3.75,4.25,4.75,5.5,6.5,7.5,9,11]
t_ranges = [0.045,0.12,0.175,0.25,0.35,0.5,0.8,1.25,1.75,2.5,4,7.5]
#xb_ranges = [0.1,0.15,0.2,0.25,0.3]
#q2_ranges = [1.0,1.5,2.0]



def img_from_pdf(img_dir):
	image_files = []
	lists = os.listdir(img_dir)
	sort_list = sorted(lists)
	for img_file in sort_list:
		print("On file " + img_file)
		image1 = Image.open(img_dir+img_file)
		image_files.append(image1)

	return image_files



def append_images(images, xb_counter, direction='horizontal', 
                  bg_color=(255,255,255), aligment='center',text_title="Title"):
    
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
        new_width = sum(widths)+max(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)

    if direction=='horizontal':
        new_im = Image.new('RGB', (int(new_width+0), int(new_height+images[0].size[1]/2)), color=bg_color)

    if xb_counter == -77:
        draw = ImageDraw.Draw(new_im)

        text = text_title
        textwidth, textheight = images[0].size[0]/1.1, images[0].size[1]/1.1

        margin = 10
        #x = images[0].size[0] - textwidth - margin
        #y = images[0].size[1] - textheight - margin
        x =  int(new_width/2)
        y = int(new_height/2)
        fonts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')
        font = ImageFont.truetype(os.path.join(fonts_path, 'agane_bold.ttf'), 150)
        draw.text((x, y), text,(0,0,0),font=font)

    offset = images[0].size[0]
    for im_counter,im in enumerate(images):
        ic(im_counter)
        if direction=='horizontal' and xb_counter>-1:
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1])/2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            print("xb counter is {}".format(xb_counter))
            if direction=='horizontal' and xb_counter == -1:
            #here we create vertical strip of Q2 values
                offset += im.size[1]

                draw = ImageDraw.Draw(new_im)

                text = str(xb_ranges[im_counter])
                textwidth, textheight = images[0].size[0]/5, images[0].size[1]/5

                margin = 10
                #x = images[0].size[0] - textwidth - margin
                #y = images[0].size[1] - textheight - margin
                x =  (int(images[0].size[0])*((1.5+im_counter)))
                ic(im_counter)
                y = int(.2*images[0].size[1])
                ic(y)
                ic(text)
                fonts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')
                font = ImageFont.truetype(os.path.join(fonts_path, 'agane_bold.ttf'), 150)


                draw.text((x, y), text,(0,0,0),font=font)

                if im_counter == 5:
                    draw = ImageDraw.Draw(new_im)

                    text = "x_B"
                    textwidth, textheight = images[0].size[0]/3, images[0].size[1]/3

                    margin = 10
                    #x = images[0].size[0] - textwidth - margin
                    #y = images[0].size[1] - textheight - margin
                    y = int(.5*images[0].size[1])
                    ic(y)
                    ic(text)
                    fonts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')
                    font = ImageFont.truetype(os.path.join(fonts_path, 'agane_bold.ttf'), 150)

                    draw.text((x, y), text,(0,0,0),font=font)


            if direction=='horizontal' and xb_counter == -77:
            #here we create vertical strip of Q2 values
                print("did nothing")
            
            elif direction=='vertical':
                x = 0
                offset = int(images[0].size[1]*im_counter/1.5)
                ic(aligment)
                if aligment == 'center':
                    x = int((new_width - im.size[0])/2)
                elif aligment == 'right':
                    x = new_width - im.size[0]
                ic(x,offset)
                new_im.paste(im, (x, offset))

            offset += im.size[1]

    if (direction=='horizontal' and xb_counter>-1):
        #This is for the x-axis labels (xB)

        draw = ImageDraw.Draw(new_im)

        text = str(q2_ranges[xb_counter])
        textwidth, textheight = images[0].size[0]/5, images[0].size[1]/5
        ic(text)
        margin = 10
        #x = images[0].size[0] - textwidth - margin
        #y = images[0].size[1] - textheight - margin
        x = 0.65*int(images[0].size[0])
        y =  0.5*int(images[0].size[1])
        #y = int(images[0].size[1])*(len(q2_ranges)-1)
        ic(x,y)
        fonts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')
        font = ImageFont.truetype(os.path.join(fonts_path, 'agane_bold.ttf'), 150)


        draw.text((x,y), text,(0,0,0),font=font)

        if xb_counter == 6:
                    draw = ImageDraw.Draw(new_im)

                    text = "Q^2"
                    textwidth, textheight = images[0].size[0]/3, images[0].size[1]/3

                    margin = 10
                    #x = images[0].size[0] - textwidth - margin
                    #y = images[0].size[1] - textheight - margin
                    x = .5*x
                    y = int(.5*images[0].size[1])
                    ic(y)
                    ic(text)
                    fonts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fonts')
                    font = ImageFont.truetype(os.path.join(fonts_path, 'agane_bold.ttf'), 150)

                    draw.text((x, y), text,(0,0,0),font=font)


    return new_im


def chunks(l, n):
	spits = (l[i:i+n] for i in range(0, len(l), n))
	return spits



#img_dir = "test_t_dep/"
#img_dir = "t_dependence_plots/"

plot_dirs = ['t_0040',  't_0170',  't_0350',  't_0800',  't_1750',  't_4000',
't_0120',  't_0250',  't_0500',  't_1250',  't_2500',  't_7500']

t_names = ['0.045','0.175','0.35','0.80','1.75','4.0','0.125','0.25','0.50','1.25','2.50','7.50']

for t_count,t_dir in enumerate(plot_dirs):
    img_dir = "../allplots/"+t_dir+"/"

    t_title = "Event Distribution Across t = {} GeV^2".format(t_names[t_count])
    ic(t_title)

    images = img_from_pdf(img_dir)
    ic(len(images))

    print(len(images))
    #print(images)
    layers = []

    num_ver_slices = len(xb_ranges)
    num_hori_slices = len(q2_ranges)
    ic(num_ver_slices)
    ic(num_hori_slices)

    #for i in range(0,int(len(images)/num_ver_slices)):
    for i in range(0,num_hori_slices):
        #print("on step "+str(i))
        layer = list(reversed(list(reversed(images[i*num_ver_slices:i*num_ver_slices+num_ver_slices]))))
        #print(layer)
        #list(reversed(array))
        ic(layer)
        layers.append(layer)

    #print(layers[0])

    horimg = []


    imglay1 = append_images(layers[0], -1, direction='horizontal')
    imglay1.save("testlay.jpg",optimize=True, quality=100)
    horimg.append(imglay1)

    #make horizontal axis labels
    #ic(len(layers[0]))

    for q2_counter,layer in enumerate(layers):
        print("len of layers is {}".format(len(layer)))
        print("counter is {}".format(q2_counter))
        print("On vertical layer {}".format(q2_counter))
        #print(layer)
        imglay = append_images(layer, q2_counter, direction='horizontal')
        imglay.save("testing1_{}.jpg".format(q2_counter))
        #sys.exit()
        horimg.append(imglay)

    imglay1 = append_images(layers[0], -77, direction='horizontal',text_title=t_title)
    imglay1.save("testlay.jpg",optimize=True, quality=100)
    horimg.append(imglay1)

    print("Joining images horizontally")
    horimg = list(reversed(horimg))
    final = append_images(horimg, 1,  direction='vertical')
    final_name = "finished_pics/"+t_title.replace("^","").replace(" ","").replace(".","").replace("=","")+".jpg"
    final.save(final_name,optimize=True, quality=100)
    print("saved {}".format(final_name))



