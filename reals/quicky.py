import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os, subprocess
import math
import shutil
from icecream import ic
import pandas as pd

def plot_1dhist(x_data,vars,y_data=[0,1],ranges="none",second_x="none",logger=False,first_label="rad",second_label="norad",
            saveplot=False,pics_dir="none",plot_title="none",first_color="blue",sci_on=False):
    
    plot_title = plot_title
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "20"

    # Initalize parameters
    x_name = vars[0]

    if ranges=="none":
        xmin = 0.99*min(x_data)
        xmax =  1.01*max(x_data)
        num_xbins = 50#int(len(x_data)/)
    else:
        xmin = ranges[0]
        xmax =  ranges[1]

    #x_bins = np.linspace(xmin, xmax, num_xbins) 

    # Creating plot
    fig, ax = plt.subplots(figsize =(14, 10)) 
        
    ax.set_xlabel(x_name)  
    ax.set_ylabel('Corrected N$_{events}$')  
    #ax.set_ylabel('NoRad$_{Gen}$/Rad$_{Rec}$')

    a = first_label
    a2 = second_label
    b = "rad"
    b2="norad"
    plt.bar(x_data,y_data,width=16,color="purple")#, range=[xmin,xmax], color='blue')
    #plt.hist(x_data, bins =x_bins, range=[xmin,xmax], color='blue', alpha=0.5, label=a)# cmap = plt.cm.nipy_spectral) 
    #if second_x is not "none":
    #    print("printing second histo")
    #    plt.hist(second_x, bins =x_bins, range=[xmin,xmax],color='red', alpha=0.5, label=a2)# cmap = plt.cm.nipy_spectral) 


    plt.legend()
    #plt.tight_layout()  

    if logger:
        plt.yscale('log')

    #Generate plot title
    if plot_title == "none":
        plot_title = 'Corrected N$_{events}$, F18 Inbending, $1.5<Q^2<2, 0.2<x_B<0.25, 0.2<-t<0.3$'
    
    plt.title(plot_title) 
    
    if sci_on:
        plt.ticklabel_format(axis="x",style="sci",scilimits=(0,0))


    if saveplot:
        new_plot_title = plot_title.replace("/","").replace(" ","_").replace("$","").replace("^","").replace("\\","").replace(".","").replace("<","").replace(">","")
        print(new_plot_title)
        

        #print(pics_dir)
        if not os.path.exists(pics_dir):
            os.makedirs(pics_dir)

        plt.savefig(pics_dir + new_plot_title+".png")
        #plt.savefig(pics_dir + new_plot_title+"_linear_scale"+".png")

        plt.close()
    else:
        plt.show()






df0 = pd.read_pickle("F18_All_DVPi0_Events_counted.pkl")

df0.to_csv("F18_csv.csv")
ic(df0.sum(axis=0))

ic(df0["ave_t"].unique())


for x in [0.175,0.225,0.275,0.34]:
    for t in [0.045,0.12,0.25,0.35]:
        print("hi")

for x in [0.34]:
    for t in [0.045,0.12,0.25,0.35]:
        print(x,t)
        df = df0.query("ave_q == 1.25 and ave_x < 0.36 and ave_x>0.32 and ave_t == {}".format(t))
        ic(df)
        y_data = df['counts']
        x_data = df['ave_p']
        err=0
        ranges = [0,360]
        vars = ["$\phi$"]

        plot_1dhist(x_data,vars,y_data=y_data,ranges=ranges,second_x="none",logger=False,first_label="rad",second_label="norad",
                saveplot=False,pics_dir="none",plot_title="none",first_color="blue",sci_on=False)

sys.exit()

corr = [0,
0,
0,
0,
0,
0,
39.14566613,
16.07751239,
17.34022556,
5.674934726,
8.088520408,
14.70430443,
28.20094192,
111.8882488,
0,
0,
0,
0,
0,
0]

radcorr = y_data*corr
ic(radcorr)
plot_1dhist(x_data,vars,y_data=radcorr,ranges=ranges,second_x="none",logger=False,first_label="rad",second_label="norad",
            saveplot=False,pics_dir="none",plot_title="none",first_color="blue",sci_on=False)
