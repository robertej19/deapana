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
from convert_root_to_pickle import convert_GEN_NORAD_root_to_pkl
from convert_root_to_pickle import convert_GEN_RAD_root_to_pkl
from convert_root_to_pickle import convert_RECON_NORAD_root_to_pkl
from convert_root_to_pickle import convert_RECON_RAD_root_to_pkl
import pickle_analysis

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


dpath = "Norad/Gen/"
#dir_name_after_cuts = '/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/After_Cuts/'
dir_name_before_cuts = '/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/Before_Cuts/'


fname = "5000_20210731_2317_norad_recon_recon_generated_events.pkl"
#fname = "5000_20210731_2317_norad_gen_all_generated_events.pkl"
#5000_20210731_2317_norad_recon_recon_generated_events.pkl

dirname = dir_name_before_cuts+dpath
# jobs_list = []
# for f in os.listdir(dirname):
#     f_path = dirname + f
#     print(f)
#     if os.path.isfile(f_path):
#         jobs_list.append(f)
# jobs_list = sorted(jobs_list)

df = pd.read_pickle(dirname+fname)
ic(df)



# if __name__ == "__main__":

#     data_paths = ["Rad/Gen/","Rad/Recon/","Norad/Gen/","Norad/Recon/",]
#     converters = [convert_GEN_RAD_root_to_pkl,convert_RECON_RAD_root_to_pkl,
#                     convert_GEN_NORAD_root_to_pkl,convert_RECON_NORAD_root_to_pkl]
#     data_types = ["Gen","Recon","Gen","Recon"]



#     for index,dpath in enumerate(data_paths):
#         data_step = 1
#         datatype = data_types[index]


#         #print("On path {}".format(dpath))
#         dirname = '/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/Raw_Root_Files/'+dpath
#         dir_name_before_cuts = '/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/Before_Cuts/'+dpath
#         dir_name_after_cuts = '/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/After_Cuts/'+dpath


#         # First we convert from Root to Pkl

#         if data_step == 0:

#             jobs_list = []
#             for f in os.listdir(dirname):
#                 f_path = dirname + f
#                 if os.path.isfile(f_path):
#                     jobs_list.append(f)
#             jobs_list = sorted(jobs_list)
#             print("Found {} files in jobs directory".format(len(jobs_list)))

#             for fname in jobs_list:
#                 print("Processing file {}".format(fname))
#                 tree = converters[index].readFile(dirname+fname)

#                 outname = fname.split(".")[0]
#                 if "Recon" in dpath:
#                     df_gen, df_rec = converters[index].readEPGG(tree)
#                     df_rec.to_pickle(dir_name_before_cuts+outname+"_reconstructed_events.pkl")
#                     df_gen.to_pickle(dir_name_before_cuts+"../Gen/"+outname+"_recon_generated_events.pkl")
#                 elif "Gen" in dpath:
#                     df_gen = converters[index].readEPGG(tree)
#                     df_gen.to_pickle(dir_name_before_cuts+outname+"_all_generated_events.pkl")

#                     print(df_gen.shape)
#                     print(df_gen.columns)
#                     print(df_gen.head(5))
#             data_step += 1
        
#         # Now we implement DVEP cuts
#         #print("Now implementing DVEP cuts")
#         if data_step == 1:
#             jobs_list = []
#             for f in os.listdir(dir_name_before_cuts):
#                 f_path = dir_name_before_cuts + f
#                 #print(f_path)
#                 if os.path.isfile(f_path):
#                     jobs_list.append(f)
#                     print(f)
#             jobs_list = sorted(jobs_list)

#             size_gen_chunks = 4000000 #Change this, depending on how big the gen dataset is

#             dfs = []

#             for fname in jobs_list:
#                 ic(fname)
#                 datatype = 'Gen' if 'gen' in fname else 'Recon'
#                 df = pd.read_pickle(dir_name_before_cuts+fname)
#                 n = size_gen_chunks  #chunk row size
#                 #ic(df.shape)
#                 list_df = []
#                 for i in range(0,df.shape[0],n):
#                     #ic(i)
#                     #ic(i+n)
#                     list_df.append(df[i:i+n])

#                 for index, df_chunk in enumerate(list_df):
#                     #print("On DF chunk {}".format(index))


#                     if datatype == "Gen":
#                         df_gen = pickle_analysis.makeGenDVpi0vars(df_chunk)
#                         df_gen.to_pickle(dir_name_after_cuts+fname)
#                         dfs.append(df_gen)
#                     elif datatype == "Recon":
#                         df_recon_pi0vars = pickle_analysis.makeDVpi0vars(df_chunk)
#                         df_recon = pickle_analysis.cutDVpi(df_recon_pi0vars)
#                         df_recon.to_pickle(dir_name_after_cuts+fname)
#                         dfs.append(df_recon)


#                 df = dfs[0]

#                 histo_plotting.make_all_histos(df,datatype=datatype,
#                     hists_2d=True,hists_1d=True,hists_overlap=False,
#                     saveplots=True,output_dir=dir_name_after_cuts+fname.split(".")[0]+"/")
