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


    a = 0
    if a == 0:
        #
        # df_after_cuts = pd.read_pickle("recon/df_recon_with_pi0cuts.pkl")
        #df_after_cuts = pd.read_pickle("recon/df_recon_all_9628_files.pkl")
        df_after_cuts = pd.read_pickle("recon_pi0_with_dups.pkl")
        
        
        #ic(df_real)

        #make plots after cuts are applied
        #ic(df_after_cuts.head(5))
        ic(df_after_cuts)
        #hist = df_after_cuts.hist(bins=30)
        #plt.show()
        a = df_after_cuts.event.values

        ic(a)
        b = np.bincount(a)
        ic(type(a))
        ic(b)
        print(b[0:10])

        c,d = np.histogram(b,bins=9,range=(0,9))
        ic(c)
        ic(d)

        c = c[1:]
        d = d[1:-1]

        fig, ax = plt.subplots(figsize =(10, 7)) 
        plt.rcParams["font.size"] = "16"
        ax.set_xlabel("Number of Combinations")  
        ax.set_ylabel("Number of Events")
        plt.title('Number Combinations Passing DVPiP Cuts Per Event, Sim. Recon')
        ax.bar(d,c)#,'ro')
        ax.set_yscale('log')
        #ax.set_xscale('log')
        plt.xlim([0.5,9.5])
        #plt.ylim([0,5])
        #plt.colorbar()
        #print(cprimes)
        print(c)
        plt.show()

        sys.exit()

        x_data = b
        x_bins = np.arange(600)
        xmin = .5
        xmax = 600

        ic(np.unique(b))
        plt.hist(x_data, bins =x_bins, range=[xmin,xmax], label='Raw Counts')
        plt.show()

        primes = []
        ii = 0
        indi = []
        for i in range(30):
            ii += i
            primes.append(ii)
            indi.append(i+1)
            
        #REMOVE 0, span down to 600

        cprimes = []
        for p in primes:
            cprimes.append(c[p])
            
        # n_p_each = c
        # x_bins_each = 
        
        # inds_to_delete = sorted(primes, reverse=True)

        # for ind in inds_to_delete:
        #     np.delete(n_p_each,ind)

        ic(primes)

        # fig, ax = plt.subplots(figsize =(10, 7)) 
        # plt.rcParams["font.size"] = "16"
        # ax.set_xlabel("Number of 2 Photon Combinations")  
        # ax.set_ylabel("Number of Events")
        # plt.title('Number of 2 Photon Combinations per Event')
        # ax.plot(x_bins,c,'+')
        # ax.plot(primes,cprimes,'ro')
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        # plt.xlim([.5,600])
        # #plt.ylim([0,5])
        # #plt.colorbar()

        # plt.show()

        # from math import comb

        # primes,cprimes

        # bases = []

        # for 
        # comb(x,2)



        fig, ax = plt.subplots(figsize =(10, 7)) 
        plt.rcParams["font.size"] = "16"
        ax.set_xlabel("Number of Photons")  
        ax.set_ylabel("Number of Events")
        plt.title('Number of Photons per Event before DVPiP Cuts, Sim. Recon')
        ax.bar(indi,cprimes)#,'ro')
        ax.set_yscale('log')
        #ax.set_xscale('log')
        plt.xlim([1.5,30])
        #plt.ylim([0,5])
        #plt.colorbar()
        print(cprimes)
        plt.show()