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


# # # dirname = "smallbinana/noradrec/"

# # # jobs_list = []
# # # for f in os.listdir(dirname):
# # #     f_path = dirname + f
# # #     if os.path.isfile(f_path):
# # #             jobs_list.append(f)

# # # #print(jobs_list)

# # # dfs = []

# # # for fname in jobs_list:
# # #     df = pd.read_pickle(dirname+fname)
# # #     dfs.append(df)


# # # df0 = pd.concat(dfs, axis=0)

# # # ic(df0)

# # # df0.to_pickle("smallbinana/noradrec_all.pkl")


normi_fac = 6.324107651E-01
plot_dir="plotsmallbins/"



fname = "smallbinana/noradrec_all.pkl"

df_norad = pd.read_pickle(fname)


fname = "smallbinana/radrec_all.pkl"

df_rad = pd.read_pickle(fname)


df_norad = df_norad.sample(int(df_norad.shape[0]*normi_fac))



#########Q2
x_data_rad = df_rad['Q2']
x_data_norad = df_norad['Q2']
vars = ["$Q^2$ (GeV$^2$)"]
plot_title = "$Q^2$ for aao_(no)rad Recon."
ranges = [1,3,50]

make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,
      saveplot=True,pics_dir=plot_dir,plot_title=plot_title ,first_color="blue",sci_on=False)



#########xb
x_data_rad = df_rad['xB']
x_data_norad = df_norad['xB']
vars = ["$x_B$"]
plot_title = "$x_B$ for aao_(no)rad Recon."
ranges = [0.15,.35,50]

make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,
      saveplot=True,pics_dir=plot_dir,plot_title=plot_title ,first_color="blue",sci_on=False)



#########Q2
x_data_rad = df_rad['t']
x_data_norad = df_norad['t']
vars = ["-t (GeV$^2$)"]
plot_title = "-t for aao_(no)rad Recon."
ranges = [0.15,.35,50]

make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,
      saveplot=True,pics_dir=plot_dir,plot_title=plot_title ,first_color="blue",sci_on=False)


#########
x_data_rad = df_rad['W']
x_data_norad = df_norad['W']
vars = ["W (GeV)"]
plot_title = "W for aao_(no)rad Recon."
ranges = [1,5,50]

make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,
      saveplot=True,pics_dir=plot_dir,plot_title=plot_title ,first_color="blue",sci_on=False)


#########Q2
x_data_rad = df_rad['phi1']
x_data_norad = df_norad['phi1']
vars = ["$\phi$ (degrees))"]
plot_title = "$\phi$ for aao_(no)rad Recon."
ranges = [100,150,50]

make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,
      saveplot=True,pics_dir=plot_dir,plot_title=plot_title ,first_color="blue",sci_on=False)


################################## GEN


fname = "smallbinana/noradgen_all.pkl"

df_norad = pd.read_pickle(fname)


fname = "smallbinana/radgen_all.pkl"

df_rad = pd.read_pickle(fname)


df_norad = df_norad.sample(int(df_norad.shape[0]*normi_fac))



#########Q2
x_data_rad = df_rad['GenQ2']
x_data_norad = df_norad['GenQ2']
vars = ["$Q^2$ (GeV$^2$)"]
plot_title = "$Q^2$ for aao_(no)rad Generators"
ranges = [1,3,50]

make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,
      saveplot=True,pics_dir=plot_dir,plot_title=plot_title ,first_color="blue",sci_on=False)



#########xb
x_data_rad = df_rad['GenxB']
x_data_norad = df_norad['GenxB']
vars = ["$x_B$"]
plot_title = "$x_B$ for aao_(no)rad Generators"
ranges = [0.15,.35,50]

make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,
      saveplot=True,pics_dir=plot_dir,plot_title=plot_title ,first_color="blue",sci_on=False)



#########Q2
x_data_rad = df_rad['Gent']
x_data_norad = df_norad['Gent']
vars = ["-t (GeV$^2$"]
plot_title = "-t for aao_(no)rad Generators"
ranges = [0.15,.35,50]

make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,
      saveplot=True,pics_dir=plot_dir,plot_title=plot_title ,first_color="blue",sci_on=False)


#########
x_data_rad = df_rad['GenW']
x_data_norad = df_norad['GenW']
vars = ["W (GeV)"]
plot_title = "W for aao_(no)rad Generators"
ranges = [1,5,50]

make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,
      saveplot=True,pics_dir=plot_dir,plot_title=plot_title ,first_color="blue",sci_on=False)


#########Q2
x_data_rad = df_rad['Genphi1']
x_data_norad = df_norad['Genphi1']
vars = ["$\phi$ (degrees)"]
plot_title = "$\phi$ for aao_(no)rad Generators"
ranges = [100,150,50]

make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,
      saveplot=True,pics_dir=plot_dir,plot_title=plot_title ,first_color="blue",sci_on=False)
