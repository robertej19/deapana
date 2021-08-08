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


data_paths = ["Rad/Gen/","Rad/Recon/","Norad/Gen/","Norad/Recon/"]




#dirname = '/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/Raw_Root_Files/'+dpath
#dir_name_before_cuts = '/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/Before_Cuts/'+dpath
#dir_name_after_cuts = '/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/After_Cuts/'+dpath



def get_counts(dir_name,file_name):
    config = "Fall_2018_Inbending/" if "Fall_2018_Inbending/" in dir_name else "Fall_2018_Outbending/"
    gen_type = "Norad/" if "Norad/" in dir_name else "Rad/"
    sim_type = "Recon/" if "Recon/" in dir_name else "Gen/"
    dir_base = "/mnt/d/GLOBUS/CLAS12/simulations/production/"+config+"Counted/"+gen_type+sim_type


    df = pd.read_pickle(dir_name+file_name)

    ic(df)
    ic(df.columns)
    xb_ranges_test =  [-1000,0.4,1000.0]
    q2_ranges_test =  [-1.0,5.0,15000.0]
    t_ranges_test =  [-10,1,1200]
    phi_ranges_test =  [-100000,90,180,270,360000]

    # xb_ranges_test =  [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    # q2_ranges_test =  [1,2,3,4,5,6,7,8,9,10,11]
    # t_ranges_test =  [0,.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6]
    # phi_ranges_test =  [0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360]


    xbr = xb_ranges_test
    q2r = q2_ranges_test
    t_r = t_ranges_test
    phr = phi_ranges_test

    ave_q = []
    ave_x = []
    ave_t = []
    ave_p = []
    counts = []


    for qmin,qmax in zip(q2r,q2r[1:]):
        print("On Q2 range: {} to {}".format(qmin,qmax))
        for xmin,xmax in zip(xbr,xbr[1:]):
            print("On xB range: {} to {}".format(xmin,xmax))
            for tmin,tmax in zip(t_r,t_r[1:]):
                for pmin,pmax in zip(phr,phr[1:]):
                    #print(qmin,qmax,xmin,xmax,tmin,tmax,pmin,pmax)
                    if sim_type == "Gen/":
                        q_statement = "GenQ2 >= {} and GenQ2 < {} and GenxB >= {} and GenxB < {} and Gent >= {} and Gent < {} and Genphi1 >= {} and Genphi1 < {}".format(qmin,qmax,xmin,xmax,tmin,tmax,pmin,pmax)
                    else:
                        q_statement = "Q2 >= {} and Q2 < {} and xB >= {} and xB < {} and t >= {} and t < {} and phi1 >= {} and phi1 < {}".format(qmin,qmax,xmin,xmax,tmin,tmax,pmin,pmax)
                        
                    df_b = df.query(q_statement)
                    count = df_b.shape[0]
                    ave_q.append((qmax+qmin)/2)
                    ave_x.append((xmax+xmin)/2)
                    ave_t.append((tmax+tmin)/2)
                    ave_p.append((pmax+pmin)/2)
                    counts.append(count)

    df_counts = pd.DataFrame(list(zip(ave_q,ave_x,ave_t,ave_p,counts)),columns=['ave_q','ave_x','ave_t','ave_p','counts'])
    ic(df_counts)



    output_file_name = file_name.split(".")[0]+"_counted.pkl"

    print("Saving DF to {}".format(dir_base+output_file_name))

    df_counts.to_pickle(dir_base+output_file_name)


if __name__ == "__main__":

    dpath = "Norad/Recon/"
    dir_name_after_cuts = '/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/After_Cuts/'+dpath
    fname = "5000_20210731_2317_norad_recon_reconstructed_events.pkl"




    #get_counts(dir_name_after_cuts,fname)
    data_paths = ["Rad/Gen/","Rad/Recon/","Norad/Gen/","Norad/Recon/"]

    for index,dpath in enumerate(data_paths):
        print(dpath)
        dirname= '/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/After_Cuts/'+dpath

        jobs_list = []
        for f in os.listdir(dirname):
            f_path = dirname + f
            if os.path.isfile(f_path):
                jobs_list.append(f)

        for file_name in jobs_list:
            get_counts(dirname,file_name)
            
