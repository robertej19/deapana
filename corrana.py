import uproot
import pandas as pd
import numpy as np
import argparse
import os, sys
from icecream import ic
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import copy
from utils.utils import dot
from utils.utils import mag
from utils.utils import mag2
from utils.utils import cosTheta
from utils.utils import angle
from utils.utils import cross
from utils.utils import vecAdd
from utils.utils import pi0Energy
from utils.utils import pi0InvMass
from utils.utils import getPhi
from utils.utils import getTheta
from utils.utils import getEnergy
from utils.utils import readFile
from utils import make_histos
from utils import histo_plotting
from utils import filestruct

from utils.fitter import plotPhi_duo
from utils.fitter import fit_function
from utils.fitter import getPhiFit
matplotlib.use('Agg') 

print("at start")

base_dir ="/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/After_Cuts/Amalgamated/"

df = pd.read_pickle(base_dir+"all_comb.pkl")

ic(df)

q_vals = df['ave_q'].unique()
x_vals = df['ave_x'].unique()
t_vals = df['ave_t'].unique()

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "20"


l1 = ['NoradGenTot','NoradRecon']
l2 = ['RadGenTot','RadRecon']

for q in q_vals:
    for x in x_vals:
        for t in t_vals:
            df_2 = df.query("ave_q == {} and ave_x == {} and ave_t == {}".format(q,x,t))
            for d1,d2 in list(zip(l1,l2)):

                data_entries_1 = df_2[d1]
                data_entries_0 = df_2[d2]*5/3
                binscenters = df_2['ave_p']
                #ic(x_data)
                xmin = 0
                xmax = 360
                first_color = "red"

                fig, ax = plt.subplots(figsize =(12, 7)) 

                bar0 = ax.bar(binscenters, data_entries_1, width=18, color='red', label=d1)
            
                duo = True
                if duo:
                    bar1 = ax.bar(binscenters, data_entries_0, width=18, color='blue', alpha=0.5, label=d2)
        # fit1, = ax.plot(xspace, fit_y_data, color='darkorange', linewidth=2.5, label='Fitted function')
                plt.title("Counts in Phi, $Q^2$={:.1f}, $x_B$ = {:.1f}, t = {:.1f}".format(q,x,t))
                plt.legend()
                plt.xlabel(r'phi')
                plt.ylabel(r'Number of entries')
                plt.ticklabel_format(axis="y",style="sci",scilimits=(0,0))


                #plt.bar(x_bins,x_data,color=first_color)#plt.hist(x_data, bins =x_bins, range=[xmin,xmax], color=first_color, label='Raw Counts')# cmap = plt.cm.nipy_spectral) 
                #plt.show()
                outname = "{}_v_{}_{:.1f}_{:.1f}_{:.1f}.pdf".format(d1,d2,q,x,t)
                plt.savefig(base_dir + "plots/"+outname)
                plt.close()
                #ic(df_2)

