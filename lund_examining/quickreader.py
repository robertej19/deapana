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
#import utils.make_histos
import make_histos



df = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/Test/panda_lunds/test_noradlund.lund.pkl")


df_e = df.query("particleID == 11")
ic(df_e.columns)
x_data = df_e['mom_x']
y_data = df_e['mom_y']
var_names = ['xb','q2']
ranges = [[-2,2,100],[-2,2,120]]

make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
        saveplot=False,pics_dir="none",plot_title="none",
        filename="ExamplePlot",units=["",""])
