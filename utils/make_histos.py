import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os, subprocess
import math
import shutil
from icecream import ic


def plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
            saveplot=False,pics_dir="none",plot_title="none"):
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "20"
    # Initalize parameters
    x_name = var_names[0]
    y_name = var_names[1]
    xmin = ranges[0]
    xmax =  ranges[1]
    num_xbins = ranges[2]
    ymin =  ranges[3]
    ymax =  ranges[4]
    num_ybins = ranges[5]
    x_bins = np.linspace(xmin, xmax, num_xbins) 
    y_bins = np.linspace(ymin, ymax, num_ybins) 

    # Creating plot
    fig, ax = plt.subplots(figsize =(10, 7)) 
    ax.set_xlabel(x_name)  
    ax.set_ylabel(y_name)

    plt.hist2d(x_data, y_data, bins =[x_bins, y_bins],
        range=[[xmin,xmax],[ymin,ymax]],norm=mpl.colors.LogNorm())# cmap = plt.cm.nipy_spectral) 

    # Adding color bar 
    if colorbar:
        plt.colorbar()

    #plt.tight_layout()  

    
    #Generate plot title
    if plot_title == "none":
        plot_title = '{} vs {}'.format(x_name,y_name)
    
    plt.title(plot_title) 
        
    
    if saveplot:
        #plot_title.replace("/","")
        new_plot_title = plot_title.replace("/","").replace(" ","_").replace("$","").replace("^","").replace("\\","").replace(".","").replace("<","").replace(">","")
        print(new_plot_title)
        plt.savefig(pics_dir + new_plot_title+".png")
        plt.close()
    else:
        plt.show()

def plot_1dhist(x_data,vars,ranges,second_x="none",
            saveplot=False,pics_dir="none",plot_title="none",first_color="red",sci_on=False):
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "20"

    # Initalize parameters
    x_name = vars[0]
    xmin = ranges[0]
    xmax =  ranges[1]
    num_xbins = ranges[2]
    x_bins = np.linspace(xmin, xmax, num_xbins) 

    # Creating plot
    fig, ax = plt.subplots(figsize =(10, 7)) 
        
    ax.set_xlabel(x_name)  
    ax.set_ylabel('counts')  
    
    
    plt.hist(x_data, bins =x_bins, range=[xmin,xmax], color=first_color, label='Raw Counts')# cmap = plt.cm.nipy_spectral) 
    if second_x is not "none":
        print("printing second histo")
        plt.hist(second_x, bins =x_bins, range=[xmin,xmax],color='black', label='With Acceptance Corr.')# cmap = plt.cm.nipy_spectral) 


    #plt.tight_layout()  


    #Generate plot title
    if plot_title == "none":
        plot_title = '{} counts'.format(x_name)
    
    plt.title(plot_title) 
    
    if sci_on:
        plt.ticklabel_format(axis="x",style="sci",scilimits=(0,0))


    if saveplot:
        new_plot_title = plot_title.replace("/","").replace(" ","_").replace("$","").replace("^","").replace("\\","").replace(".","").replace("<","").replace(">","")
        plt.savefig(pics_dir + new_plot_title+".png")
        plt.close()
    else:
        plt.show()



if __name__ == "__main__":
    ranges = [0,1,100,0,300,120]
    variables = ['xB','Phi']
    conditions = "none"
    datafile = "F18In_168_20210129/skims-168.pkl"
    plot_2dhist(datafile,variables,ranges)