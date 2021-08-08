import numpy as np; np.random.seed(42)
import matplotlib.pyplot as plt
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
import matplotlib as mpl


from utils.fitter import plotPhi_duo
from utils.fitter import fit_function
from utils.fitter import getPhiFit


dirname = "/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/After_Cuts/Norad/Gen/"
fname = "5000_20210731_2317_norad_gen_all_generated_events_1.pkl"
fname = "5000_20210731_2317_norad_recon_recon_generated_events_0.pkl"


df = pd.read_pickle(dirname+fname)
ic(df.shape)

# two input arrays
azimut = df['Genphi1']
radius = df['GenPtheta']

ic(azimut.shape)
# define binning
rbins = np.linspace(0,radius.max(), 4)
abins = np.linspace(0,2*np.pi, 4)

#calculate histogram
hist, _, _ = np.histogram2d(azimut, radius, bins=(abins, rbins))
A, R = np.meshgrid(abins, rbins)

# plot
fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))

pc = ax.pcolormesh(A, R, hist.T, norm=mpl.colors.LogNorm())#cmap='hsv'); cmap="magma_r")
fig.colorbar(pc)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate random data:
N = 10000
#r = .5 + np.random.normal(size=N, scale=.1)
#theta = np.pi / 2 + np.random.normal(size=N, scale=.1)
theta = df['Genphi1']
r = df['GenPtheta']

# Histogramming
nr = 50
ntheta = 200
r_edges = np.linspace(0, 1, nr + 1)
theta_edges = np.linspace(0, 2*np.pi, ntheta + 1)
H, _, _ = np.histogram2d(r, theta, [r_edges, theta_edges])

# Plot
ax = plt.subplot(111, polar=True)
Theta, R = np.meshgrid(theta_edges, r_edges)
ax.pcolormesh(Theta, R, H)
plt.show()



# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from physt import special

# # # Generate some points in the Cartesian coordinates
# # np.random.seed(42)

# # gen = lambda l, h, s = 3000: np.asarray([random.random() * (h - l) + l for _ in range(s)])

# # X = gen(-100, 100)
# # Y = gen(-1000, 1000)
# # Z = gen(0, 1400)


# X = df['Genphi1']
# Y = df['GenPtheta']


# hist = special.polar_histogram(X, Y, weights=Z, radial_bins=40)
# # ax = hist.plot.polar_map()

# hist.plot.polar_map(density=True, show_zero=False, cmap="inferno", lw=0.5, figsize=(5, 5))
# plt.show()
