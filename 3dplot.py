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
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os, subprocess
import math
import shutil
from icecream import ic

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


df = pd.read_pickle("this_is_test_data.pkl")

fig = plt.figure()
ax = plt.axes(projection="3d")

z_points = df['ave_x']
x_points = df['ave_q']
y_points = df['ave_p']
scalers = df['counts']/df['counts'].max()

ax.scatter3D(x_points, y_points, z_points, s=4000*scalers,c=df['counts'], norm=mpl.colors.LogNorm())#cmap='hsv');

plt.show()