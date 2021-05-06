#!/usr/bin/env python3
"""
A simple script to save Z and X of 6862 nflow project.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import argparse
import sys 
import pandas as pd
from matplotlib.patches import Rectangle

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

Ebeam = 10.6 #GeV, CLAS12 beam energy
m_p = 0.938 #GeV, proton mass
alpha = 1/137 #, fine structure constant


c6 = pd.read_csv("raw-clas6-data.csv")
c6f = c6.query("q2 == 2.24 or q2 == 2.25")

c6f.loc[:,"omega"] = c6f.loc[:,"q2"] / (2*m_p*c6f.loc[:,"xb"])
c6f.loc[:,"y"] = c6f.loc[:,"omega"] / Ebeam
c6f.loc[:,"epsilon"] = (1 - c6f.loc[:,"y"] - c6f.loc[:,"q2"]/(4*Ebeam*Ebeam))/(1 - c6f.loc[:,"y"] +  c6f.loc[:,"y"]*c6f.loc[:,"y"]/2 + c6f.loc[:,"q2"]/(4*Ebeam*Ebeam))
c6f.loc[:,"gamma"] =  alpha/(8*np.pi)*c6f.loc[:,"q2"]/(m_p**2*Ebeam**2)*(1-c6f.loc[:,"xb"])/(c6f.loc[:,"xb"]**3)*(1/(1-c6f.loc[:,"epsilon"]))

c6f.loc[:,"A"] = c6f.loc[:,"tel"] *c6f.loc[:,"gamma"] /(2*np.pi)
c6f.loc[:,"B"] = c6f.loc[:,"tt"] *c6f.loc[:,"gamma"] *c6f.loc[:,"epsilon"]/(2*np.pi)
c6f.loc[:,"C"] = c6f.loc[:,"lt"] *c6f.loc[:,"gamma"] *np.sqrt(2*c6f.loc[:,"epsilon"]*(1+c6f.loc[:,"epsilon"]))/(2*np.pi)


abc = c6f.query("t==0.25")

A = abc["A"].values[0]
B = abc["B"].values[0]
C = abc["C"].values[0]




def fit_function(phi,A,B,C):
    #A + B*np.cos(2*phi) +C*np.cos(phi)
    rads = phi*np.pi/180
    #return (A * np.exp(-x/beta) + B * np.exp(-1.0 * (x - mu)**2 / (2 * sigma**2)))
    #A = T+L, B=TT, C=LT
    #A = black, B=blue, C=red
    return A + B*np.cos(2*rads) + C*np.cos(rads)

def getPhiFit(keep_bins,realbins,phi_vals,phi_title,plot_dir,saveplot=False,sci_on=True,log_on=False):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "20"
    
    xmin = 0
    xmax = 360
    #print("fitting {}".format(phi_title))
    
    #data = phi_vals
    bins_x = np.linspace(xmin, xmax, 21)
    #data_entries, bins = np.histogram(data,bins=bins_x)
    ic(phi_vals)
    data_entries = phi_vals.to_numpy()
    ic(data_entries)
    bins = bins_x
    data_errors = np.sqrt(data_entries)
    data_errors = [1/err if err>0 else err+1 for err in data_errors]
    
    ic(data_entries)
    ic(data_errors)

    

    if 1==1:
        bins = realbins
        ic(bins)
        
        #binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        binscenters = bins+9
        ic(binscenters)
        ic(data_entries)
        ic("TRYING TO RUN")
        ind_to_drop = np.where(keep_bins<1)
        ic("TRYING TO RUN2")
        ic(ind_to_drop)
        #binscenters = np.delete(binscenters, ind_to_drop)
        # data_entries = np.delete(data_entries, ind_to_drop)
        # data_errors = np.delete(data_errors, ind_to_drop)
        # ic(binscenters)
        # ic(data_entries)
        # ic(keep_bins)

        
        # 5.) Fit the function to the histogram data.
        popt, pcov = curve_fit(fit_function, xdata=binscenters, ydata=data_entries, p0=[2.0, 2, 0.3],
                    sigma=data_errors, absolute_sigma=True)
        #print(popt) #popt contains the values for A, B, C

        a_err = np.sqrt(pcov[0][0])
        b_err = np.sqrt(pcov[1][1])
        c_err = np.sqrt(pcov[2][2])

        a,b,c = popt[0],popt[1],popt[2]
        #ic(a_err,b_err,c_err)
        #ic.disable()
        
        # 6.)
        # Generate enough x values to make the curves look smooth.
       
        kopt = (A,B,C)

        fit_CLAS6_data_1 = fit_function(binscenters, *kopt)

        ic(binscenters)

        ic(fit_CLAS6_data_1)

    
        

        # chisq0 = stats.chisquare(f_obs=data_entries, f_exp=fit_y_data_1)
        # chisq = stats.chisquare(f_obs=np.array(data_entries, dtype=np.float64), f_exp=np.array(fit_y_data_1, dtype=np.float64))

        # sums=[]
        # for ind,val in enumerate(fit_y_data_1):
        #     diff2 = (data_entries[ind]-val)**2
        #     s1 = diff2/val
        #     sums.append(s1)

        # manchisq = np.sum(sums)

        # ###ic.enable()
        # if chisq0[0]<0:
        #     ic(manchisq)
        #     ic(chisq0[0])
        # if not (chisq0[0] == chisq[0]):
        #     print("ERROR MISMATCH")
        #     print(chisq0[0])
        #     print(chisq[0])
        #     print(manchisq)

        # ic.disable()

        # p = chisq[1]
        # chisq = chisq[0]

        # ic(chisq)
        # ic(p)


        xspace = np.linspace(binscenters.min()-9,binscenters.max()+9, 1000)

        #fit_y_data = fit_function(xspace, *popt)
        fit_y_data = fit_function(xspace, *kopt)
        

        ###ic.enable()
        ic(fit_y_data)
        
        # y_manual = []
        # for ind, val in enumerate(xspace):
        #     ic(val,a,b,c)
        #     y_one = fit_function(val,a,b,c)
        #     ic(y_one)
        #     y_manual.append(y_one)


        
        #7
        # Plot the histogram and the fitted function.

        fig, ax = plt.subplots(figsize =(12, 7)) 


        corrfact = fit_CLAS6_data_1/data_entries
        ic(corrfact)

        corrfact = [i if i<5000 else 0 for i in corrfact]
        ic(corrfact)
        bar1 = ax.bar(binscenters, corrfact, width=bins[1] - bins[0], color='navy', label='Histogram entries')
        

        
        
        #bar2 = ax.bar(binscenters, fit_CLAS6_data_1, width=bins[1] - bins[0], color='red', label='CLAS6 entries')
        #bar1 = ax.bar(binscenters, data_entries, width=bins[1] - bins[0], color='navy', label='Histogram entries')
        #fit1, = ax.plot(xspace, fit_y_data, color='darkorange', linewidth=2.5, label='Fitted function')

        # Make the plot nicer.
        plt.xlim(xmin,xmax)
        #plt.ylim(0,300)
        plt.xlabel(r'phi')
        plt.ylabel('Corrected $N_{events}$/Lumi/Bin Vol')

        plot_title = plot_dir + phi_title+".png"
        plt.title(phi_title)
        #plt.legend(loc='best')

        chisq = 1
        p = 1
        fit_params = "A: {:2.6f} \n B:{:2.6f} \n C:{:2.6f}".format(a,b,c)

        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        #ax.legend([bar1, fit1, extra], ("Data","A+Bcos(2Phi)+Ccos(Phi)",fit_params))
        if sci_on:
            plt.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
        if log_on:
            ax.set_yscale("log", nonposy='clip')
        #plt.text(120, max(data_entries)/1.3, fit_params)

        
        if saveplot:
            new_plot_title = plot_title.replace("/","").replace(" ","_").replace("$","").replace("^","").replace("\\","").replace(".","").replace("<","").replace(">","").replace(",","_")
            plt.savefig(plot_dir + new_plot_title+".png")
            plt.close()
            print("Saved to {}".format(plot_dir))
        else:
            plt.show()
            plt.close()
        #print("plot saved to {}".format(plot_title))

        return popt, pcov, chisq, p


def plotPhi_duo(phi_bins,bin_counts_0,bin_counts_1,phi_title,pics_dir,saveplot=False,legend=False,duo=False,fitting=False,sci_on=False):
    
    ic(phi_bins)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "20"
    
    data_entries_0 = bin_counts_0
    data_entries_1 = bin_counts_1
    bins = phi_bins

    data_errors_0 = np.sqrt(data_entries_0)
    data_errors_0 = [1/err if err>0 else err+1 for err in data_errors_0]

    data_errors_1 = np.sqrt(data_entries_1)
    data_errors_1 = [1/err if err>0 else err+1 for err in data_errors_1]

    #print(data_entries)

    if 1==1:
        ic(bins)
        binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        binscenters = np.append(binscenters,np.array([351,]),axis=0)
        ic(binscenters)
        # 5.) Fit the function to the histogram data.
        #popt, pcov = curve_fit(fit_function, xdata=binscenters, ydata=data_entries, p0=[2.0, 2, 0.3],
         #           sigma=data_errors, absolute_sigma=True)
        #print(popt) #popt contains the values for A, B, C

        ##a_err = np.sqrt(pcov[0][0])
        ##b_err = np.sqrt(pcov[1][1])
        #c_err = np.sqrt(pcov[2][2])

        #a,b,c = popt[0],popt[1],popt[2]
        #ic(a_err,b_err,c_err)
        #ic.disable()
        
        # 6.)
        # Generate enough x values to make the curves look smooth.
       
        #fit_y_data_1 = fit_function(binscenters, *popt)

        #ic(fit_y_data_1)

        

        #chisq0 = stats.chisquare(f_obs=data_entries, f_exp=fit_y_data_1)
        #chisq = stats.chisquare(f_obs=np.array(data_entries, dtype=np.float64), f_exp=np.array(fit_y_data_1, dtype=np.float64))

        #sums=[]
        #for ind,val in enumerate(fit_y_data_1):
        #    diff2 = (data_entries[ind]-val)**2
        #    s1 = diff2/val
        #    sums.append(s1)

       # manchisq = np.sum(sums)

        ###ic.enable()
        #if chisq0[0]<0:
        #    ic(manchisq)
        #    ic(chisq0[0])
        #if not (chisq0[0] == chisq[0]):
        #    print("ERROR MISMATCH")
        #    print(chisq0[0])
        #    print(chisq[0])
        #    print(manchisq)


       # p = chisq[1]
       # chisq = chisq[0]

        ##ic(chisq)
        #ic(p)


        #xspace = np.linspace(0, xmax, 1000)
        #fit_y_data = fit_function(xspace, *popt)

        ##ic.enable()
        #ic(fit_y_data)
        
        #y_manual = []
        #for ind, val in enumerate(xspace):
        #    ic(val,a,b,c)
        #    y_one = fit_function(val,a,b,c)
        #    ic(y_one)
        #    y_manual.append(y_one)


        
        #7
        # Plot the histogram and the fitted function.
        fig, ax = plt.subplots(figsize =(12, 7)) 
        
        


        highPower = data_entries_0
        lowPower = data_entries_1


        #plt.bar(binscenters, highPower,  
        #        color='b', label='LUND Events')
        #plt.bar(binscenters,  lowPower, color='r', alpha=0.5, label='Sim Events')




        #ic.enable()
        #ic(binscenters)
        #ic(data_entries_0)
        #ic(data_entries_1)
        bar0 = ax.bar(binscenters, data_entries_1, width=bins[1] - bins[0], color='red', label='Raw')
        
        if duo:
            bar1 = ax.bar(binscenters, data_entries_0, width=bins[1] - bins[0], color='black', label='Corrected')
       # fit1, = ax.plot(xspace, fit_y_data, color='darkorange', linewidth=2.5, label='Fitted function')


        # Make the plot nicer.
        plt.xlim(0,360)
        #plt.ylim(0,5)
        plt.xlabel(r'phi')
        plt.ylabel(r'Number of entries')

        plot_title = phi_title
        plt.title(phi_title)
        #plt.legend(loc='best')
        if legend:
            plt.legend()
        if sci_on:
            plt.ticklabel_format(axis="y",style="sci",scilimits=(0,0))

        #fit_params = "A: {:2.2f} +/- {:2.2f}\n B:{:2.2f} +/- {:2.2f}\n C:{:2.2f} +/- {:2.2f}\n Chi:{:2.2f} \n p:{:2.2f}".format(a,a_err,b,b_err,c,c_err,chisq,p)


        #plt.text(120, max(data_entries)/1.3, fit_params)

        
        if saveplot:
            new_plot_title = plot_title.replace("/","").replace(" ","_").replace("$","").replace("^","").replace("\\","").replace(".","").replace("<","").replace(">","").replace(",","_")
            plt.savefig(pics_dir + new_plot_title+".png")
            plt.close()
        else:
            plt.show()
            plt.close()
        #print("plot saved to {}".format(plot_title))





acc_cutoff = 0.005
tmin = 0.2
tmax = 0.3
lumi = 5.512e7
binvol = 0.5*0.8*18/180*np.pi*(tmax-tmin)

# final.loc[:, "acc"] = final.loc[:, "recon_counts"]/final.loc[:, "gen_counts"]
# final.loc[:, "acc_inv"] = 1/final.loc[:, "acc"]
# final.loc[:, "keep_bin"] = (np.sign(final.loc[:, "acc"]-acc_cutoff)+1)/2
# final.loc[:, "real_corr"] = final.loc[:, "real_counts"]/final.loc[:, "acc"]*final.loc[:, "keep_bin"]
# final.loc[:, "real_corr_LBV"] = final.loc[:, "real_corr"]/(lumi*binvol)
# final.loc[:, "real_corr"] =final.loc[:, "real_corr"].fillna(0)
# ic(final)
# final_0 = final.query("tmin=={}".format(tmin))

# final_0 = final_0.query("keep_bin==1")

final = pd.read_pickle("Real_phi_binned.pkl")

final.loc[:, "xsec"] = final.loc[:, "real_counts"]/(lumi*binvol)
final.loc[:, "keep_bin"] = (np.sign( final.loc[:, "real_counts"]))
final_0 = final.query("tmin=={}".format(tmin))

ic(final_0)

getPhiFit(final_0["keep_bin"],final_0.index,final_0["xsec"],"F18In: Phi Fit, {}<t<{}".format(tmin,tmax),"pics/",saveplot=False,sci_on=False,log_on=True)

sys.exit()


#plotPhi_duo(final_0.index,final_0["real_counts"],final_0["real_counts"],"raw counts, {}<t<{}".format(tmin,tmax),"pics/",legend=True)

#plotPhi_duo(final_0.index,final_0["real_corr_LBV"],final_0["real_corr_LBV"],"F18In: Corrected, Lumi, BinVol, {}<t<{}".format(tmin,tmax),"pics/",saveplot=True,sci_on=True)
#plotPhi_duo(final_0.index,final_0["real_counts"],final_0["real_corr"],"F18In: Raw vs. Corrected, {}<t<{}".format(tmin,tmax),"pics/",legend=True,duo=True,saveplot=False)
#plotPhi_duo(final_0.index,final_0["recon_counts"],final_0["gen_counts"],"Gen vs Rec, {}<t<{}".format(tmin,tmax),"pics/",legend=True,duo=True,sci_on=True,saveplot=True)
#plotPhi_duo(final_0.index,final_0["acc"],final_0["acc"],"Acc. Corr. , Low Acc Bins Removed {}<t<{}".format(tmin,tmax),"pics/",saveplot=True)
#plotPhi_duo(final_0.index,final_0["acc_inv"],final_0["acc_inv"],"Acceptance Correction","pics/",)

#plotPhi_duo(final_0.index,final_0["real_corr"],final_0["real_corr"],"real vs real corr","pics/",legend=True,fitting=True)
#getPhiFit(final_0["keep_bin"],final_0["real_corr"],"plot","pics/")

