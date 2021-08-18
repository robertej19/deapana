import pandas as pd
import numpy as np 
#import matplotlib.pyplot as plt 
import random 
import sys
import os, subprocess
from pdf2image import convert_from_path
import math
from icecream import ic
import shutil
from PIL import Image, ImageDraw, ImageFont
#This project
#import utils.make_histos
#import matplotlib
#matplotlib.use('Agg')
from utils import make_histos

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
#df_rad = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/Test/panda_lunds/test_noradlund.lund.pkl")

#df_norad = pd.read_pickle("lund_examining/evented_norad/100K_norad_example.lund.pkl_events.pkl")
#df_rad = pd.read_pickle("lund_examining/evented_rad/100K_rad_example.lund.pkl_events.pkl")
#df_norad = pd.read_pickle("lund_examining/aug2021/evented_norad/EXAMPLENORAD5000.lund.pkl_events.pkl")
#df_rad = pd.read_pickle("lund_examining/aug2021/evented_rad/EXAMPLERAD4000.lund.pkl_events.pkl")

df_norad = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/After_Cuts/Norad/Gen/5000_20210731_2317_norad_gen_all_generated_events_0.pkl")
df_rad = pd.read_pickle("/mnt/d/GLOBUS/CLAS12/simulations/production/Fall_2018_Inbending/After_Cuts/Rad/Gen/4000_20210731_2324_rad_gen_all_generated_events_0.pkl")
ic(df_rad)
#df_norad.loc[:,"W"] = np.sqrt(df_norad["W2"]) 
#df_rad.loc[:,"W"] = np.sqrt(df_rad["W2"]) 

#df_norad = df_norad.query(" W2 > 4")
#df_rad = df_rad.query(" W2 > 4")

if df_rad.shape[0]<df_norad.shape[0]:
        df_norad = df_norad.sample(df_rad.shape[0])
else:
        df_rad = df_rad.sample(df_norad.shape[0])

ic(df_rad)
ic(df_rad.head(3))

ic(df_rad.columns)



# # # # # # #M = 0.938272081 # target mass
# # # # # # M = 0.938
# # # # # # me = 0.5109989461 * 0.001 # electron mass
# # # # # # #me = 0.511
# # # # # # ebeam = 10.6 # beam energy
# # # # # # pbeam = np.sqrt(ebeam * ebeam - me * me) # beam electron momentum
# # # # # # #pbeam = ebeam
# # # # # # beam = [0, 0, pbeam] # beam vector
# # # # # # target = [0, 0, 0] # target vector
# # # # # # alpha = 1/137 #Fund const
# # # # # # mp = 0.938 #Mass proton
# # # # # # prefix = alpha/(8*np.pi)
# # # # # # E = 10.6

# # # # # # VmissPi0 = [-df_norad["GenEpx"] - df_norad["GenPpx"], -df_norad["GenEpy"] -
# # # # # #         df_norad["GenPpy"], pbeam - df_norad["GenEpz"] - df_norad["GenPpz"]]
# # # # # # df_norad.loc[:,'MM2_ep'] = np.sqrt((-M - ebeam + df_norad["GenEe"] + df_norad["GenPe"])**2 - mag2(VmissPi0))


# # # # # # VmissPi0_rad = [-df_rad["GenEpx"] - df_rad["GenPpx"], -df_rad["GenEpy"] -
# # # # # #         df_rad["GenPpy"], pbeam - df_rad["GenEpz"] - df_rad["GenPpz"]]
# # # # # # df_rad.loc[:,'MM2_ep'] = np.sqrt((-M - ebeam + df_rad["GenEe"] + df_rad["GenPe"])**2 - mag2(VmissPi0_rad))



# # # # # # # df_rad['GenG2e'] = 0
# # # # # # # df_rad['GenG2px'] = 0
# # # # # # # df_rad['GenG2py'] = 0
# # # # # # # df_rad['GenG2pz'] = 0



# # # # # # # ic(df_rad['GenEe'].head(1))
# # # # # # # ic(df_rad['GenEpx'].head(1))
# # # # # # # ic(df_rad['GenEpy'].head(1))
# # # # # # # ic(df_rad['GenEpz'].head(1))
# # # # # # # ic(df_rad['q2'].head(1))





# # # # # # gam = [df_rad['GenG2px'], df_rad['GenG2py'], df_rad['GenG2pz']]
# # # # # # df_rad.loc[:,"Etheta"] = getTheta(gam)
# # # # # # df_rad.loc[:,"Ephi"] = getPhi(gam)


# # # # # # ele = [df_rad['GenEpx'], df_rad['GenEpy'], df_rad['GenEpz']]
# # # # # # df_rad.loc[:,"Gtheta"] = getTheta(ele)
# # # # # # df_rad.loc[:,"Gphi"] = getPhi(ele)


# # # # # # df_rad.loc[:,"MissE"] = ebeam+M- df_rad["GenEe"]- df_rad["GenPe"]
# # # # # # df_rad.loc[:,"MissPx"] = -df_rad["GenEpx"]- df_rad["GenPpx"]
# # # # # # df_rad.loc[:,"MissPy"] = -df_rad["GenEpy"]- df_rad["GenPpy"]
# # # # # # df_rad.loc[:,"MissPz"] = pbeam-df_rad["GenEpz"]- df_rad["GenPpz"]


# # # # # # df_rad.loc[:,'MM2_alt'] = np.sqrt(df_rad.loc[:,"MissE"]*df_rad.loc[:,"MissE"]-df_rad.loc[:,"MissPx"]*df_rad.loc[:,"MissPx"]-df_rad.loc[:,"MissPy"]*df_rad.loc[:,"MissPy"]-df_rad.loc[:,"MissPz"]*df_rad.loc[:,"MissPz"])

# # # # # # #df_rad.loc[:,"total_E"] = df_rad['GenEe']+df_rad['GenPe']+df_rad['GenPie']+df_rad['GenGe2']
# # # # # # #df_rad.loc[:,"ang_diff"] = np.square(df_rad['GenGtheta2']-df_rad['GenEtheta'])


# # # # # # df_rad.loc[:,"ang_diff"] = df_rad['Gtheta']-df_rad['Etheta']


# # # # # # df_rad.loc[:,"log_fac"] = 1#-1*np.sign(df_rad.loc[:,"ang_diff"]-5)



# # # # # # #df_rad.loc[:,'Q2_new'] = -1*((10.6-df_rad["GenG2e"]-df_rad["GenEe"])**2-((-df_rad["GenG2px"]-df_rad["GenEpx"])**2 +(-df_rad["GenG2py"]-df_rad["GenEpy"])**2 +(pbeam-df_rad["GenG2pz"]-df_rad["GenEpz"])**2))
# # # # # # #df_rad.loc[:,'Q2_new'] = -1*((10.6-df_rad["GenEe"])**2-(0-df_rad["GenEpx"])**2 -(0-df_rad["GenEpy"])**2 -(10.6-df_rad["GenEpz"])**2)




# # # # # # VmissPi0_rad2 = [-df_rad["GenEpx"] - df_rad["GenPpx"]- df_rad["log_fac"]*df_rad["GenG2px"], -df_rad["GenEpy"] -
# # # # # #         df_rad["GenPpy"]- df_rad["log_fac"]*df_rad["GenG2py"], pbeam - df_rad["GenEpz"] - df_rad["GenPpz"]- df_rad["log_fac"]*df_rad["GenG2pz"]]
# # # # # # df_rad.loc[:,'MM2_ep_2'] = np.sqrt((-M - ebeam + df_rad["GenEe"] + df_rad["log_fac"]*df_rad["GenG2e"]+df_rad["GenPe"])**2 - mag2(VmissPi0_rad2))


plot_dir = "100k_plots_2D"
#plot_dir = "100k_plots/"


search_term = " GenxB > 0.2 and GenxB<0.25 and Genphi1 > 108 and Genphi1 < 126 and GenQ2 > 1.5 and GenQ2 < 2"
df_rad0 = df_rad.query(search_term)
df_norad0 = df_norad.query(search_term)

x_data_rad = df_rad0['Gent']
x_data_norad = df_norad0['Gent']
vars = ["-t (GeV$^2$)"]
plot_title = "-t for aao_(no)rad Generators,  Q2=1.75 xB = 0.225 phi = 117"
ranges = [0,3,100]


make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=False,
      saveplot=False,pics_dir=plot_dir,plot_title=plot_title,first_color="blue",sci_on=False)
ic(df_rad0.shape)
ic(df_norad0.shape)


search_term = "0.2 < Gent and Gent < 0.3 and GenxB > 0.2 and GenxB<0.25 and Genphi1 > 108 and Genphi1 < 126"
df_rad0 = df_rad.query(search_term)
df_norad0 = df_norad.query(search_term)


x_data_rad = df_rad0['GenQ2']
x_data_norad = df_norad0['GenQ2']
vars = ["$Q^2$ (GeV$^2$)"]
plot_title = "$Q^2$ for aao_(no)rad Generators, t=0.25 xB = 0.225 phi = 117"
ranges = [0,6,100]

make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=False,
      saveplot=False,pics_dir=plot_dir,plot_title=plot_title,first_color="blue",sci_on=False)
ic(df_rad0.shape)
ic(df_norad0.shape)



search_term = "0.2 < Gent and Gent < 0.3 and GenxB > 0.2 and GenxB<0.25 and GenQ2 > 1.5 and GenQ2 < 2"
df_rad0 = df_rad.query(search_term)
df_norad0 = df_norad.query(search_term)

x_data_rad = df_rad0['Genphi1']
x_data_norad = df_norad0['Genphi1']
vars = ["$\phi$"]
plot_title = "$\phi$ for aao_(no)rad Generators, Q2=1.75 xB = 0.225 -t=0.25"
ranges = [0,360,36]

make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=False,
      saveplot=False,pics_dir=plot_dir,plot_title=plot_title,first_color="blue",sci_on=False)
ic(df_rad0.shape)
ic(df_norad0.shape)


search_term = "0.2 < Gent and Gent < 0.3 and Genphi1 > 108 and Genphi1 < 126 and GenQ2 > 1.5 and GenQ2 < 2"
df_rad0 = df_rad.query(search_term)
df_norad0 = df_norad.query(search_term)

x_data_rad = df_rad0['GenxB']
x_data_norad = df_norad0['GenxB']
vars = ["$x_B$"]
plot_title = "$x_B$ for aao_(no)rad Generators, Q2=1.75 t=0.25 phi = 117"
ranges = [0,1,100]

make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=False,
      saveplot=False,pics_dir=plot_dir,plot_title=plot_title,first_color="blue",sci_on=False)
ic(df_rad0.shape)
ic(df_norad0.shape)





sys.exit()


# df_rad = df_rad.query("W > 1.9")
# df_norad = df_norad.query("W > 1.9")
# ic(df_rad)
# ic(df_norad)
# if df_rad.shape[0]<df_norad.shape[0]:
#         df_norad = df_norad.sample(df_rad.shape[0])
# else:
#         df_rad = df_rad.sample(df_norad.shape[0])

# dfcsv = df_rad.head(5)
# dfcsv.to_csv("testcsv.csv")

#df_rad = df_rad.query("ang_diff < 8 and ang_diff >5 and q2 < 0.9")

#x_data_rad = df_rad['MM2_ep']
#x_data_rad = df_norad['MM2_ep']
#x_data_norad = df_rad['MM2_alt']

#############3#W2
# x_data_rad = df_rad['W']
# x_data_norad = df_norad['W']
# vars = ["$W (GeV)$"]
# plot_title = "$W$ for aao_(no)rad Generators"
# ranges = [1,2.1,550]

# make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=False,
#       saveplot=False,pics_dir=plot_dir,plot_title=plot_title,first_color="blue",sci_on=False)




################ #MM_ep
# # x_data_rad = df_rad['MM2_ep']
# # x_data_norad = df_norad['MM2_ep']
# # vars = ["Missing Mass (e'p) (GeV)"]
# # plot_title = "MM$_{e'p}$ for aao_(no)rad Generators"
# # ranges = [0,1,100]

# # make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,
# #       saveplot=True,pics_dir=plot_dir,plot_title=plot_title+ ", W > 1.9 GeV",first_color="blue",sci_on=False)


# # #sys.exit()

# # # ################Radiated energy
# # # x_data_rad = df_rad['GenG2e']
# # # x_data_norad =  "none"
# # # vars = ["Radiated Photon Energy (GeV)"]
# # # plot_title = "Energy of Radiated Photon, aao_rad"
# # # ranges = [0,3,100]

# # # make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,
# # #       saveplot=False,pics_dir=plot_dir,plot_title=plot_title,first_color="blue",sci_on=False)



# # # #TOTAL ENERGY:
# # x_data_norad = df_norad["GenEe"]+ df_norad["GenPe"]+ df_norad["GenGe"]+ df_norad["GenG2e"]
# # x_data_rad = df_rad["GenEe"]+ df_rad["GenPe"]+ df_rad["GenPie"]+ df_rad["GenG2e"]
# # vars = ["Total Energy (GeV)"]
# # plot_title = "Total Energy for aao_(no)rad Generators"
# # ranges = [6,12,100]

# # make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,
# #       saveplot=True,pics_dir=plot_dir,plot_title=plot_title + ", W > 1.9 GeV",first_color="blue",sci_on=False)


# # ####Theta G
# # x_data_rad = df_rad['Gtheta']-df_rad['Etheta']
# # #x_data_norad = df_rad['Gtheta']-df_rad['Etheta']
# # x_data_norad="none"
# # vars = ["$\\theta_{\gamma} - \\theta_{e}$"]
# # plot_title = "Photon-Electron Angle (theta) difference"
# # ranges = "none"
# # ranges = [-5,15,200]

# # make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,
# #       saveplot=True,pics_dir=plot_dir,plot_title=plot_title + ", W > 1.9 GeV",first_color="blue",sci_on=False)


# # #########Q2


# x_data_rad = df_rad['q2']
# x_data_norad = df_norad['q2']
# vars = ["$Q^2$ (GeV$^2$)"]
# plot_title = "$Q^2$ for aao_(no)rad Generators"
# ranges = [0,12,100]

# make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,
#       saveplot=False,pics_dir=plot_dir,plot_title=plot_title + ", W > 1.9 GeV",first_color="blue",sci_on=False)
# sys.exit()

# # # #xB
# # x_data_rad = df_rad['xb']
# # x_data_norad = df_norad['xb']
# # vars = ["$x_B$"]
# # plot_title = "$x_B$ for aao_(no)rad Generators"
# # ranges = [0,1,100]

# # make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,
# #       saveplot=True,pics_dir=plot_dir,plot_title=plot_title + ", W > 1.9 GeV",first_color="blue",sci_on=False)





# # ###########3 #MM_ep - only norad
# # x_data_rad = df_rad['MM2_ep']
# # x_data_norad = df_norad['MM2_ep']
# # vars = ["Missing Mass (e'p) (GeV)"]
# # plot_title = "MM$_{e'p}$ for only aao_norad Generator"
# # ranges = [0.13,0.15,100]
# # x_data_rad = x_data_norad
# # x_data_norad = "none"

# # make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,first_label="norad",
# #       saveplot=True,pics_dir=plot_dir,plot_title=plot_title + ", W > 1.9 GeV",first_color="blue",sci_on=False)




# # ###########3 #MM_ep - only rad
# # x_data_rad = df_rad['MM2_ep']
# # x_data_norad = df_rad['MM2_ep_2']
# # vars = ["Missing Mass (e'p) (GeV)"]
# # plot_title = "MM$_{e'p}$ and MM$_{e'p\gamma}$ for only aao_rad Generator"
# # ranges = [0,1,100]
# # first_label="(e+p-e'-p')$^2$"
# # second_label = "(e+p-(e'+$\gamma$)-p')$^2$"

# # make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,first_label=first_label,second_label = second_label,
# #       saveplot=True,pics_dir=plot_dir,plot_title=plot_title + ", W > 1.9 GeV",first_color="blue",sci_on=False)




# # ############Ee
# # x_data_rad = df_rad['GenEe']
# # x_data_norad = df_norad['GenEe']
# # vars = ["Electron Energy (GeV)"]
# # plot_title = "Electron Energy Distribution for aao_(no)rad Generators"
# # ranges = [0,11,100]

# # make_histos.plot_1dhist(x_data_rad,vars,ranges=ranges,second_x=x_data_norad,logger=True,
# #       saveplot=True,pics_dir=plot_dir,plot_title=plot_title + ", W > 1.9 GeV",first_color="blue",sci_on=False)



# # sys.exit()

ic(x_data_rad)
#ic(x_data_norad)



# x_data_rad = df_rad['GenEe']+df_rad['GenPe']+df_rad['GenPie']+df_rad['GenGe2']
# #x_data_rad = df_rad['ang_diff']
# ic(df_norad['MM2_ep'])

# #x_data_norad = x_data_rad
# x_data_norad = df_norad['GenEe']+df_norad['GenPe']+df_norad['GenGe']+df_norad['GenGe2']

# ### Theta diff vs Q2
# x_data_rad = df_rad['Gtheta']-df_rad['Etheta']
# y_data_rad = df_rad['Q2_alt']
# var_names = ["$\\theta_{\gamma} - \\theta_{e}$","$Q^2$ (GeV$^2$)"]
# ranges = [[-10,20,200],[0,12,200]]
# plot_title = "Electron Photon Angle Diff. vs. $Q^2$"

# # rad E vs Q2
# # x_data_rad = df_rad['GenG2e']
# # y_data_rad = df_rad['q2']
# # var_names = ["Radiated Photon Energy","Q$^2$ (GeV$^2$)"]
# # ranges = [[0,12,100],[0,1,200]]
# # plot_title = "5<$\\theta_{\gamma} - \\theta_{e}$<8, $Q^2$<0.9"


# # W vs Ee
# x_data_rad = df_rad['W']
# y_data_rad = df_rad['GenG2e']
# var_names = ["W","Electron Energy"]
# ranges = [[1,4,100],[0,2,200]]
# plot_title = "W vs Ee rad"



# ### angle diff vs total energy
# y_data_rad = df_rad["GenEe"]+ df_rad["GenPe"]+ df_rad["GenPie"]+ df_rad["GenG2e"]
# x_data_rad = df_rad['ang_diff']
# var_names = ["Photon Electron Angle Difference","Total Energy"]
# ranges = [[-10,20,200],[7,12,200]]
# plot_title = "Photon Electron Angle Difference vs. Total Energy"


### angle diff vs total energy
y_data_rad = df_rad["GenEe"]+ df_rad["GenPe"]+ df_rad["GenPie"]+ df_rad["GenG2e"]
x_data_rad = df_rad['ang_diff']
var_names = ["Photon Electron Angle Difference","Total Energy"]
ranges = [[-10,20,200],[7,12,200]]
plot_title = "Photon Electron Angle Difference vs. Total Energy"





make_histos.plot_2dhist(x_data_rad,y_data_rad,var_names,ranges,colorbar=True,
            saveplot=False,pics_dir="100k_plots/",plot_title=plot_title,
            filename="ElectronVPhoton",units=["GeV","GeV"])

#ranges = [11.52,11.56,100]





sys.exit()

df_e = df.query("particleID == 11")
ic(df_e.columns)
x_data = df_e['mom_x']
y_data = df_e['mom_y']
var_names = ['xb','q2']
ranges = [[-2,2,100],[-2,2,120]]

make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
        saveplot=False,pics_dir="none",plot_title="none",
        filename="ExamplePlot",units=["",""])
