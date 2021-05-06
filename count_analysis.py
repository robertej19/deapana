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
from utils import histo_plotting
from utils import filestruct

from utils.fitter import plotPhi_duo
from utils.fitter import fit_function
from utils.fitter import getPhiFit
fs = filestruct.fs()
data_dir = "data/binned/"

parser = argparse.ArgumentParser(description="Get args",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d","--dirname", help="a directory of pickle files", default=data_dir)
parser.add_argument("-m","--merge", help="merge all pkl files", default = False, action='store_true')
parser.add_argument("-p","--plot", help="merge all pkl files", default = False, action='store_true')
args = parser.parse_args()

if args.merge:
    files = os.listdir(args.dirname)

    print(files)
    dfs = []
    dfs_gen = []

    for f in files:
        df = pd.read_pickle(args.dirname+f)
        if "Gen" in f:
            dfs_gen.append(df)
        elif "merged" in f:
            pass
        else:
            dfs.append(df)
        
    results = pd.concat(dfs,axis='columns')
    results = results.loc[:,~results.columns.duplicated()]

    results_gen = pd.concat(dfs_gen,axis=1)
    results_gen = results_gen.loc[:,~results_gen.columns.duplicated()]

    results_gen['total_gen_count'] = results_gen[[col for col in results_gen.columns if col.startswith('gen_counts')]].sum(axis=1)

    rfinal = pd.merge(results,  results_gen)
    cols = [c for c in rfinal.columns if c.lower()[:3] != 'gen']
    rfinal = rfinal[cols]


    rfinal.loc[:,"q2_mean"] = ((rfinal.loc[:,"q2_min"]+rfinal.loc[:,"q2_max"])/2) 
    rfinal.loc[:,"xb_mean"] = ((rfinal.loc[:,"xb_min"]+rfinal.loc[:,"xb_max"])/2) 
    rfinal.loc[:,"t_mean"] = ((rfinal.loc[:,"tmin"]+rfinal.loc[:,"t_max"])/2) 
    rfinal.loc[:,"phi_mean"] = ((rfinal.loc[:,"phi_min"]+9)) 

    rfinal.loc[:,"omega1"] = rfinal.loc[:, "q2_mean"] / (2*fs.m_p*rfinal.loc[:,"xb_mean"])
    rfinal.loc[:,"y1"] = rfinal.loc[:,"omega1"] / fs.Ebeam
    rfinal.loc[:,"epsilon1"] = (1 - rfinal.loc[:,"y1"] - rfinal.loc[:,"q2_mean"]/(4*fs.Ebeam*fs.Ebeam))/(1 - rfinal.loc[:,"y1"] +  rfinal.loc[:,"y1"]*rfinal.loc[:,"y1"]/2 + rfinal.loc[:,"q2_mean"]/(4*fs.Ebeam*fs.Ebeam))
    rfinal.loc[:,"gamma1"] =  fs.alpha/(8*np.pi)*rfinal.loc[:,"q2_mean"]/(fs.m_p**2*fs.Ebeam**2)*(1-rfinal.loc[:,"xb_mean"])/(rfinal.loc[:,"xb_mean"]**3)*(1/(1-rfinal.loc[:,"epsilon1"]))


    rfinal.loc[:,"binvol"] = (rfinal.loc[:,"q2_max"]-rfinal.loc[:,"q2_min"])*(rfinal.loc[:,"t_max"]-rfinal.loc[:,"tmin"])*(rfinal.loc[:,"xb_max"]-rfinal.loc[:,"xb_min"])*(18*np.pi/180)

    

    rfinal["acc_corr"] = rfinal["total_gen_count"]/rfinal["recon_counts"]
    rfinal["xsec_nocorr"] = rfinal["real_counts"]/(rfinal["binvol"]*fs.f18_inbending_total_lumi_inv_nb)
    rfinal["xsec_withcorr"] = rfinal["xsec_nocorr"]*rfinal["acc_corr"]
    #*(2*np.pi)/rfinal["gamma1"]



    ic(rfinal)
    
    df_out = rfinal[['gamma1','epsilon1','q2_mean', 'xb_mean', 't_mean','phi_mean','xsec_nocorr','acc_corr','xsec_withcorr']].copy()
    ic(df_out)

    outname = args.dirname+"final_count.pkl"
    ic(outname)
    df_out.to_pickle(outname)

if args.plot:

    inname = args.dirname+"final_count.pkl"
    df = pd.read_pickle(inname)
    ic(df)

    t = 0.25
    df = df.query("t_mean == {}".format(t))
    ic(df)
    #plotPhi_duo(df['phi_mean'],df['xsec_withcorr'],df['xsec_withcorr'],"title",'./',saveplot=False,legend=False,duo=False,fitting=False,sci_on=False)
    #getPhiFit(df['acc_corr'].values,df['phi_mean'].values,df['xsec_withcorr'].values,"title",'./',saveplot=False,sci_on=False)
    
    c6 = pd.read_csv("data/raw-clas6-data.csv")


    c6f = c6.query("q2 == 2.24 or q2 == 2.25")
    ic(c6f)
    c6f.loc[:,"omega"] = c6f.loc[:,"q2"] / (2*fs.m_p*c6f.loc[:,"xb"])
    c6f.loc[:,"y"] = c6f.loc[:,"omega"] / fs.Ebeam
    c6f.loc[:,"epsilon"] = (1 - c6f.loc[:,"y"] - c6f.loc[:,"q2"]/(4*fs.Ebeam*fs.Ebeam))/(1 - c6f.loc[:,"y"] +  c6f.loc[:,"y"]*c6f.loc[:,"y"]/2 + c6f.loc[:,"q2"]/(4*fs.Ebeam*fs.Ebeam))
    c6f.loc[:,"gamma"] =  fs.alpha/(8*np.pi)*c6f.loc[:,"q2"]/(fs.m_p**2*fs.Ebeam**2)*(1-c6f.loc[:,"xb"])/(c6f.loc[:,"xb"]**3)*(1/(1-c6f.loc[:,"epsilon"]))

    c6f.loc[:,"A"] = c6f.loc[:,"tel"] *c6f.loc[:,"gamma"] /(2*np.pi)
    c6f.loc[:,"B"] = c6f.loc[:,"tt"] *c6f.loc[:,"gamma"] *c6f.loc[:,"epsilon"]/(2*np.pi)
    c6f.loc[:,"C"] = c6f.loc[:,"lt"] *c6f.loc[:,"gamma"] *np.sqrt(2*c6f.loc[:,"epsilon"]*(1+c6f.loc[:,"epsilon"]))/(2*np.pi)


    abc = c6f.query("t==0.25")

    A = abc["A"].values[0]
    B = abc["B"].values[0]
    C = abc["C"].values[0]

    kopt = (A,B,C)

    getPhiFit(df['acc_corr'].values,df['phi_mean'].values,df['xsec_withcorr'].values,"Cross Section, t={}GeV^2, xb=0.34, Q2 = 2.25".format(t),'./',saveplot=False,sci_on=False,kopt=kopt)






    #OLD:

    # plotPhi_duo(final_0.index,final_0["real_counts"],final_0["real_counts"],"raw counts, {}<t<{}".format(tmin,tmax),"pics/",legend=True)

    # #plotPhi_duo(final_0.index,final_0["real_corr_LBV"],final_0["real_corr_LBV"],"F18In: Corrected, Lumi, BinVol, {}<t<{}".format(tmin,tmax),"pics/",saveplot=True,sci_on=True)
    # plotPhi_duo(final_0.index,final_0["real_counts"],final_0["real_corr"],"F18In: Raw vs. Corrected, {}<t<{}".format(tmin,tmax),"pics/",legend=True,duo=True,saveplot=False)
    # #plotPhi_duo(final_0.index,final_0["recon_counts"],final_0["gen_counts"],"Gen vs Rec, {}<t<{}".format(tmin,tmax),"pics/",legend=True,duo=True,sci_on=True,saveplot=True)
    # #plotPhi_duo(final_0.index,final_0["acc"],final_0["acc"],"Acc. Corr. , Low Acc Bins Removed {}<t<{}".format(tmin,tmax),"pics/",saveplot=True)
    # #plotPhi_duo(final_0.index,final_0["acc_inv"],final_0["acc_inv"],"Acceptance Correction","pics/",)

    # #plotPhi_duo(final_0.index,final_0["real_corr"],final_0["real_corr"],"real vs real corr","pics/",legend=True,fitting=True)
    # #getPhiFit(final_0["keep_bin"],final_0["real_corr"],"plot","pics/")

    # getPhiFit(final_0["keep_bin"],final_0.index,final_0["real_corr_LBV"],"F18In: Phi Fit, {}<t<{}".format(tmin,tmax),"pics/",saveplot=True)


email = True
if email:
    import os
    from datetime import datetime
    from pytools import circle_emailer

    now = datetime.now()
    script_end_time = now.strftime("%H:%M:%S")
    s_name = os.path.basename(__file__)
    subject = "Completion of {}".format(s_name)
    body = "Your script {} finished running at {}".format(s_name,script_end_time)
    circle_emailer.send_email(subject,body)

