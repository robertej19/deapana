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
from convert_root_to_pickle import convert_GEN_NORAD_root_to_pkl
from convert_root_to_pickle import convert_GEN_RAD_root_to_pkl
from convert_root_to_pickle import convert_RECON_NORAD_root_to_pkl
from convert_root_to_pickle import convert_RECON_RAD_root_to_pkl
import pickle_analysis
pd.set_option('mode.chained_assignment', None)

import random 
import sys
import os, subprocess
import argparse
import shutil
import time
from datetime import datetime 
import json
M = 0.938272081 # target mass
me = 0.5109989461 * 0.001 # electron mass
ebeam = 10.604 # beam energy
pbeam = np.sqrt(ebeam * ebeam - me * me) # beam electron momentum
beam = [0, 0, pbeam] # beam vector
target = [0, 0, 0] # target vector
alpha = 1/137 #Fund const
mp = 0.938 #Mass proton
prefix = alpha/(8*np.pi)
E = 10.6


fs = filestruct.fs()

def getEnergy(vec1, mass):
    # for taken 3d momenta p and mass m, return energy = sqrt(p**2 + m**2)
    return np.sqrt(mag2(vec1)+mass**2)



if __name__ == "__main__":

    # data_paths = ["Rad/Gen/","Rad/Recon/","Norad/Gen/","Norad/Recon/",]
    # converters = [convert_GEN_RAD_root_to_pkl,convert_RECON_RAD_root_to_pkl,
    #                convert_GEN_NORAD_root_to_pkl,convert_RECON_NORAD_root_to_pkl]
    # data_paths = ["Rad/Gen/","Rad/Recon/","Norad/Gen/","Norad/Recon/"]


    #     #print("On path {}".format(dpath))
    #     dirname = '/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/Raw_Root_Files/'+dpath
    #     dir_name_after_cuts = '/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/After_Cuts/'+dpath

    #dir_name_before_cuts = '/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/Before_Cuts/Rad/Gen/'

    df_rad = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/After_Cuts/Rad/Gen/4000_20210731_2324_rad_gen_all_generated_events_0.pkl')
    df_norad = pd.read_pickle('/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/After_Cuts/Norad/Gen/5000_20210731_2317_norad_gen_all_generated_events_0.pkl')

    ic(df_rad.columns)
    ic(df_norad.columns)

    #df = df.head(30)
    #ic(df.shape)
    #ic(df2.shape)
    vars = ["p z2"]

    # ic(df.columns)
    # #df3= df.query("GenW > 2 and GenQ2 > 1")
    # #df3= df
    # #df4= df2.query("GenW > 2 and GenQ2 > 1")

    # #df4 = df4.sample(df3.shape[0])
    # ic(df3.shape)
    # ic(df4.shape)

    me = 0.1349 #pion mass
    m_pro = 0.938272081 # target mass
    m_ele = 0.5109989461 * 0.001 # electron mass
    e_beam = 10.6

    ele = [df_rad['GenPipx'], df_rad['GenPipy'], df_rad['GenPipz']]

    

    gam1 = [df_norad['GenGpx'], df_norad['GenGpy'], df_norad['GenGpz']]
    gam2 = [df_norad['GenGpx2'], df_norad['GenGpy2'], df_norad['GenGpz2']]

    df_rad.loc[:,'GenPie'] = getEnergy(ele,me)

    df_norad.loc[:,'GenPiM'] = pi0InvMass(gam1,gam2)
    df_norad.loc[:,'GenPie'] = pi0Energy(gam1,gam2)


    #df_norad.loc[:,'MMep2'] = (e_beam+m_pro-df_norad['GenEe']-df_norad['GenPe'])**2 - (e_beam-df_norad['GenEe']-df_norad['GenPp'])**2

    M = 0.938272081 # target mass
    me = 0.5109989461 * 0.001 # electron mass
    ebeam = 10.6 # beam energy
    pbeam = np.sqrt(ebeam * ebeam - me * me) # beam electron momentum
    beam = [0, 0, pbeam] # beam vector
    target = [0, 0, 0] # target vector
    alpha = 1/137 #Fund const
    mp = 0.938 #Mass proton
    prefix = alpha/(8*np.pi)
    E = 10.6


    VmissPi0 = [-df_norad["GenEpx"] - df_norad["GenPpx"], -df_norad["GenEpy"] -
                df_norad["GenPpy"], pbeam - df_norad["GenEpz"] - df_norad["GenPpz"]]
    df_norad.loc[:,'MM2_ep'] = np.sqrt((-M - ebeam + df_norad["GenEe"] + df_norad["GenPe"])**2 - mag2(VmissPi0))


    VmissPi0_rad = [-df_rad["GenEpx"] - df_rad["GenPpx"], -df_rad["GenEpy"] -
                df_rad["GenPpy"], pbeam - df_rad["GenEpz"] - df_rad["GenPpz"]]
    df_rad.loc[:,'MM2_ep'] = np.sqrt((-M - ebeam + df_rad["GenEe"] + df_rad["GenPe"])**2 - mag2(VmissPi0_rad))


    df_rad.loc[:,"total_E"] = df_rad['GenEe']+df_rad['GenPe']+df_rad['GenPie']+df_rad['GenGe2']
    df_rad.loc[:,"ang_diff"] = np.square(df_rad['GenGtheta2']-df_rad['GenEtheta'])

    df_rad = df_rad.query("ang_diff < 0.0001")

    df_rad = df_rad.query("GenW >2 and GenQ2>1")
    df_norad = df_norad.query("GenW >2 and GenQ2>1")
    df_norad = df_norad.sample(df_rad.shape[0])
    ic(df_rad.shape)
    ic(df_norad.shape)
    ic(df_rad.columns)
     
    
    x_data_rad = df_rad['GenEe']+df_rad['GenPe']+df_rad['GenPie']+df_rad['GenGe2']
    #x_data_rad = df_rad['ang_diff']
    ic(df_norad['MM2_ep'])
    #x_data_rad = df_rad['GenQ2']
    #x_data_norad = df_norad['GenQ2']
    #x_data_norad = x_data_rad
    x_data_norad = df_norad['GenEe']+df_norad['GenPe']+df_norad['GenGe']+df_norad['GenGe2']

    # x_data_rad = df_rad['GenEe']
    # #x_data_norad = df_norad['GenEe']
    # y_data_rad = df_rad['GenGe2']
    # #x_data = [1,1,1,2,1,1,1,1,1,1,1,1,1,2,2,1,2,1,2,1,5]
    # #x2data = [3,3,3,3,3,3,4,4,4,4,4,4,3,3,2,1,2,1,2,1]

    vars = ["Total Energy (GeV)"]
    #var_names = ["Energy, Electron","Energy, Rad. Photon"]
    #ranges = [[0,12,100],[0,3.5,100]]
    #x_data_rad =  df_rad['GenEe']+df_rad['GenPe']+df_rad['GenGe2']+df_rad['GenPie']
    #y_data_rad = df_rad['GenGe2']
    #make_histos.plot_2dhist(x_data_rad,y_data_rad,var_names,ranges,colorbar=True,
    #            saveplot=True,pics_dir="./",plot_title="2DVERSUS",
    #            filename="ElectronVPhoton",units=["GeV","GeV"])
    #ranges = [1,11,100]
    ranges = "none"
    make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,
            saveplot=True,pics_dir="./",plot_title="subset2",first_color="blue",sci_on=False)


        # Total energy (e'+p+pi0+radiattive photon for aaa_rad.)
        # Q2=-(e'-e)^2
        # W=sqrt[(e'-e)+target proton)]^2
        # xB
        # Missing mass^2 =[(e+p)-(e'+p')]^2.  Expect pi0 mass^2 for nomad and slightly different for rad.

    sys.exit()
    # df3.loc[:,'NetE'] = df3['GenEe']+df3['GenGe2']


    # x_data = df3['NetE']
    
    # make_histos.plot_1dhist(x_data,vars,ranges="none",second_x='none',
    #         saveplot=True,pics_dir="radplots/",plot_title="NETENERGY",first_color="blue",sci_on=False)




    # sys.exit()
    #ic(df["GenGpy2"])
    #ic(df["GenGpx2"])

    varies = ['GenEpx', 'GenEpy', 'GenEpz', 'GenEp', 'GenEtheta', 'GenEphi',
                       'GenPpx', 'GenPpy', 'GenPpz', 'GenPp', 'GenPtheta', 'GenPphi',
                       'GenEe',
                       'GenPe','GenQ2',
                       'Gennu', 'GenxB', 'Gent', 'GenW', 'GenMPt', 'Genphi1', 'Genphi2']
    for var in varies: 
        #var = "GenQ2"
        x_data = df3[var]
        x2dat = df4[var]
        vars = [str(var.split("en")[1])+" (GeV)"]
        #ranges = [0,3,100]
        make_histos.plot_1dhist(x_data,vars,ranges="none",second_x=x2dat,
                saveplot=True,pics_dir="radplots/",plot_title=var,first_color="blue",sci_on=False)
    
