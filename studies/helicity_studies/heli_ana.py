#!/usr/bin/env python3
"""
A simple script to save Z and X of 6862 nflow project.
"""

import uproot
import pandas as pd
import numpy as np
import argparse
import os, sys
from icecream import ic
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


# 1.) Necessary imports.    
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import argparse
import sys 
import pandas as pd
from matplotlib.patches import Rectangle

from icecream import ic

# 2.) Define fit function.
def fit_function(phi,A):
    #A + B*np.cos(2*phi) +C*np.cos(phi)
    rads = phi*np.pi/180
    #return (A * np.exp(-x/beta) + B * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))
    #A = T+L, B=TT, C=LT
    #A = black, B=blue, C=red
    return A*np.sin(rads)


M = 0.938272081 # target mass
me = 0.5109989461 * 0.001 # electron mass
ebeam = 10.604 # beam energy
pbeam = np.sqrt(ebeam * ebeam - me * me) # beam electron momentum
beam = [0, 0, pbeam] # beam vector
target = [0, 0, 0] # target vector
   
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get args",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-f","--fname", help="a single root file to convert into pickles", default="infile.root")
    parser.add_argument("-o","--out", help="a single pickle file name as an output", default="outfile.pkl")
    parser.add_argument("-s","--entry_stop", help="entry_stop to stop reading the root file", default = None)
    
    args = parser.parse_args()

    # t ranges:
    # 0.2 - 0.3, .4, .6, 1.0
    # xb ranges:
    # 0.25, 0.3, 0.38
    # q2 ranges:
    # 3, 3.5, 4, 4.5

    # #xxxxxxxxxxxxxxxxxxxxxxxxxxx
    # #Push though the REAL data
    # #xxxxxxxxxxxxxxxxxxxxxxxxxxx
    #df_real = pd.read_pickle("df_real.pkl")
    #df_after_cuts = pd.read_pickle("df_real_5000_5099.pkl")

    q2bins = [[2,2.56],[2.56,3.8],[3.8,4.8],[4.8,6],[6,11]]
    t1 =  [[0, 0.39], [0.39, 0.71], [0.71, 2]]
    t2 =  [[0, 0.47], [0.47, 0.87], [0.87, 2] ]
    t3 =  [[0, 0.57], [0.57, 0.99], [0.99, 2]]
    t4 = [[0, 0.67], [0.67, 1.09], [1.09, 2] ]
    t5 = [[0, 0.91], [0.91, 1.37], [1.37, 2]]
    tbins = [t1,t2,t3,t4,t5]    
    phi_bins = [[0,40],[40,80],[80,120],[120,160],[160,200],[200,240],[240,280],[280,320],[320,360]]


    a = 1
    if a == 0:
        df_after_cuts = pd.read_pickle("real/F18_All_DVPi0_Events.pkl")
        #ic(df_real)

        #make plots after cuts are applied
        ic(df_after_cuts.head(5))
        #hist = df_after_cuts.hist(bins=30)
        #plt.show()
        ic(df_after_cuts.columns.values)


        def get_counts(q2min,q2max,tmin,tmax,phimin,phimax,df):
            cut_q = "Q2>{} & Q2<{} & t>{} & t<{} & phi1>{} & phi1<{}".format(q2min,q2max,tmin,tmax,phimin,phimax)
            df_small_gen = df.query(cut_q)
            return df_small_gen
            

        

        
        qmins = []
        qmaxs = []
        tmins = []
        tmaxs = []
        phimins = []
        phimaxs = []
        Nposs = []
        Nnegs = []

        for q2range,tbins in zip(q2bins,tbins):
            q2min = q2range[0]
            q2max = q2range[1]
            for trange in tbins:
                tmin = trange[0]
                tmax = trange[1]
                for phirange in phi_bins:
                    phimin = phirange[0]
                    phimax = phirange[1]
                    df_small = get_counts(q2min,q2max,tmin,tmax,phimin,phimax,df_after_cuts)
                    #ic(df_small)
                    num_pos = len(df_small.query("helicity == -1")) #Sign flip because jlab is stupid
                    #ic(df_pos)
                    num_neg = len(df_small.query("helicity == 1")) #Sign flip because jlab is stupid
                    qmins.append(q2min)
                    qmaxs.append(q2max)
                    tmins.append(tmin)
                    tmaxs.append(tmax)
                    phimins.append(phimin)
                    phimaxs.append(phimax)
                    Nposs.append(num_pos)
                    Nnegs.append(num_neg)
        
        print(len(Nnegs))

        df = pd.DataFrame([qmins,qmaxs,tmins,tmaxs,phimins,phimaxs,Nposs,Nnegs]).transpose()
        df.columns =['q2min','q2max','tmin','tmax','phimin','phimax','Npos','Nneg']
        df['phimean'] = (df.phimin + df.phimax)/2

        df.to_pickle("heli.pkl")
        ic(df)    
        sys.exit()

    else:

        df = pd.read_pickle("heli.pkl")
        df["Q"] = df.Npos - df.Nneg
        df["P"] = df.Npos + df.Nneg
        df["dq"] = np.sqrt(df.Npos+df.Nneg)

        df["assym"] = df.Q/df.P
        df["assym_err"] = np.abs(df.assym)*np.sqrt( np.square(df.dq/df.Q)+np.square(df.dq/df.P) )
        df["assym_err"] = df["assym_err"].fillna(0.03)
    
        ic(df)


        def get_counts(q2min,q2max,tmin,tmax,df):
            cut_q = "q2min=={} & q2max=={} & tmin=={} & tmax=={}".format(q2min,q2max,tmin,tmax)
            df_small_gen = df.query(cut_q)
            return df_small_gen

        hardcounter = 0
        for q2range,tbins in zip(q2bins,tbins):
            q2min = q2range[0]
            q2max = q2range[1]
            for tind,trange in enumerate(tbins):
                tmin = trange[0]
                tmax = trange[1]
                df_small = get_counts(q2min,q2max,tmin,tmax,df)

                ic(df_small)
                x = df_small.phimean.values
                y = df_small.assym.values/.86
            

                fig, ax = plt.subplots(figsize =(10, 7)) 
                plt.rcParams["font.size"] = "16"
                ax.set_xlabel("Phi")  
                ax.set_ylabel("Assymetry")
                plt.title('Assymetry, F18in, Q2: {} - {} GeV2, t: {} - {} GeV2'.format(q2min,q2max,tmin,tmax))
                plt.errorbar(x,y,yerr=df_small.assym_err.values, fmt='o',marker='s', mfc='red',
                    ms=10)#s=100, c='b', marker='o',yerr=df_small.assym_err.values)
                plt.ylim([-.2,.2])

                # plt.show(block=False)
                # plt.pause(.9)
                # plt.close()
                # sys.exit()
                popt, pcov = curve_fit(fit_function, xdata=x, ydata=y, p0=[1],
                    sigma=df_small.assym_err.values, absolute_sigma=True)

                print(popt)
                popt, pcov = curve_fit(fit_function, xdata=x, ydata=y, p0=[popt[0]],
                    sigma=df_small.assym_err.values, absolute_sigma=True)
                print(popt)

                
                xmax = 360
                xspace = np.linspace(0, xmax, 1000)
                fit_y_data = fit_function(xspace, *popt)
                fit1, = ax.plot(xspace, fit_y_data, color='darkorange', linewidth=2.5, label='Fitted function')

                #plt.show(block=False)
                #plt.pause(1.2)

                q2minp = round(q2min*100,0)#str(q2min).replace(".","")
                tminp = str(round(tmin*100,0))
                tmaxp = round(tmax*100,0)
                if len(tminp)<2:
                    tminp = "00"+tminp
                if len(tminp)<3:
                    tminp = "0" +tminp
                #hardc = str(hardcounter)
                #if len(hardc)<2:
                #    hardc = "0"+hardc
                plot_title = 'Assymetryt_{}_q2_{}_{}.png'.format(tind,q2minp,q2max)
                plt.savefig('assympics/'+plot_title)
                #plt.pause(1.2)
                hardcounter +=1
                plt.close()




        
    #df_small_gen = df_after_cuts
    