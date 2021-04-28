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


M = 0.938272081 # target mass
me = 0.5109989461 * 0.001 # electron mass
ebeam = 10.604 # beam energy
pbeam = np.sqrt(ebeam * ebeam - me * me) # beam electron momentum
beam = [0, 0, pbeam] # beam vector
target = [0, 0, 0] # target vector
   
def makeGenDVpi0vars(df_epgg):

    # useful objects
    ele = [df_epgg['GenEpx'], df_epgg['GenEpy'], df_epgg['GenEpz']]
    df_epgg.loc[:, 'GenEp'] = mag(ele)
    df_epgg.loc[:, 'GenEe'] = getEnergy(ele, me)
    df_epgg.loc[:, 'GenEtheta'] = getTheta(ele)
    df_epgg.loc[:, 'GenEphi'] = getPhi(ele)

    pro = [df_epgg['GenPpx'], df_epgg['GenPpy'], df_epgg['GenPpz']]
    df_epgg.loc[:, 'GenPp'] = mag(pro)
    df_epgg.loc[:, 'GenPe'] = getEnergy(pro, M)
    df_epgg.loc[:, 'GenPtheta'] = getTheta(pro)
    df_epgg.loc[:, 'GenPphi'] = getPhi(pro)

    gam = [df_epgg['GenGpx'], df_epgg['GenGpy'], df_epgg['GenGpz']]
    df_epgg.loc[:, 'GenGp'] = mag(gam)
    df_epgg.loc[:, 'GenGe'] = getEnergy(gam, 0)
    df_epgg.loc[:, 'GenGtheta'] = getTheta(gam)
    df_epgg.loc[:, 'GenGphi'] = getPhi(gam)

    gam2 = [df_epgg['GenGpx2'], df_epgg['GenGpy2'], df_epgg['GenGpz2']]
    df_epgg.loc[:, 'GenGp2'] = mag(gam2)
    df_epgg.loc[:,'GenGe2'] = getEnergy(gam2, 0)
    df_epgg.loc[:, 'GenGtheta2'] = getTheta(gam2)
    df_epgg.loc[:, 'GenGphi2'] = getPhi(gam2)

    print(1)
    pi0 = vecAdd(gam, gam2)
    VGS = [-df_epgg['GenEpx'], -df_epgg['GenEpy'], pbeam - df_epgg['GenEpz']]
    v3l = cross(beam, ele)
    v3h = cross(pro, VGS)
    v3g = cross(VGS, gam)
    VmissPi0 = [-df_epgg["GenEpx"] - df_epgg["GenPpx"], -df_epgg["GenEpy"] -
                df_epgg["GenPpy"], pbeam - df_epgg["GenEpz"] - df_epgg["GenPpz"]]
    VmissP = [-df_epgg["GenEpx"] - df_epgg["GenGpx"] - df_epgg["GenGpx2"], -df_epgg["GenEpy"] -
                df_epgg["GenGpy"] - df_epgg["GenGpy2"], pbeam - df_epgg["GenEpz"] - df_epgg["GenGpz"] - df_epgg["GenGpz2"]]
    Vmiss = [-df_epgg["GenEpx"] - df_epgg["GenPpx"] - df_epgg["GenGpx"] - df_epgg["GenGpx2"],
                -df_epgg["GenEpy"] - df_epgg["GenPpy"] - df_epgg["GenGpy"] - df_epgg["GenGpy2"],
                pbeam - df_epgg["GenEpz"] - df_epgg["GenPpz"] - df_epgg["GenGpz"] - df_epgg["GenGpz2"]]

    df_epgg.loc[:, 'GenMpx'], df_epgg.loc[:, 'GenMpy'], df_epgg.loc[:, 'GenMpz'] = Vmiss

    print(2)
    # binning kinematics
    df_epgg.loc[:,'GenQ2'] = -((ebeam - df_epgg['GenEe'])**2 - mag2(VGS))
    df_epgg.loc[:,'Gennu'] = (ebeam - df_epgg['GenEe'])
    df_epgg.loc[:,'GenxB'] = df_epgg['GenQ2'] / 2.0 / M / df_epgg['Gennu']
    df_epgg.loc[:,'Gent'] = 2 * M * (df_epgg['GenPe'] - M)
    df_epgg.loc[:,'GenW'] = np.sqrt(np.maximum(0, (ebeam + M - df_epgg['GenEe'])**2 - mag2(VGS)))
    df_epgg.loc[:,'GenMPt'] = np.sqrt((df_epgg["GenEpx"] + df_epgg["GenPpx"] + df_epgg["GenGpx"] + df_epgg["GenGpx2"])**2 +
                                (df_epgg["GenEpy"] + df_epgg["GenPpy"] + df_epgg["GenGpy"] + df_epgg["GenGpy2"])**2)

    print(3)
    # trento angles
    df_epgg['Genphi1'] = angle(v3l, v3h)
    df_epgg['Genphi1'] = np.where(dot(v3l, pro) > 0, 360.0 -
                                df_epgg['Genphi1'], df_epgg['Genphi1'])
    df_epgg['Genphi2'] = angle(v3l, v3g)
    df_epgg['Genphi2'] = np.where(dot(VGS, cross(v3l, v3g)) <
                                0, 360.0 - df_epgg['Genphi2'], df_epgg['Genphi2'])

    print(4)

    return df_epgg

def makeDVpi0vars(df_epgg):

    # useful objects
    ele = [df_epgg['Epx'], df_epgg['Epy'], df_epgg['Epz']]
    df_epgg.loc[:, 'Ep'] = mag(ele)
    df_epgg.loc[:, 'Ee'] = getEnergy(ele, me)
    df_epgg.loc[:, 'Etheta'] = getTheta(ele)
    df_epgg.loc[:, 'Ephi'] = getPhi(ele)

    pro = [df_epgg['Ppx'], df_epgg['Ppy'], df_epgg['Ppz']]
    df_epgg.loc[:, 'Pp'] = mag(pro)
    df_epgg.loc[:, 'Pe'] = getEnergy(pro, M)
    df_epgg.loc[:, 'Ptheta'] = getTheta(pro)
    df_epgg.loc[:, 'Pphi'] = getPhi(pro)

    gam = [df_epgg['Gpx'], df_epgg['Gpy'], df_epgg['Gpz']]
    df_epgg.loc[:, 'Gp'] = mag(gam)
    df_epgg.loc[:, 'Ge'] = getEnergy(gam, 0)
    df_epgg.loc[:, 'Gtheta'] = getTheta(gam)
    df_epgg.loc[:, 'Gphi'] = getPhi(gam)

    gam2 = [df_epgg['Gpx2'], df_epgg['Gpy2'], df_epgg['Gpz2']]
    df_epgg.loc[:, 'Gp2'] = mag(gam2)
    df_epgg.loc[:,'Ge2'] = getEnergy(gam2, 0)
    df_epgg.loc[:, 'Gtheta2'] = getTheta(gam2)
    df_epgg.loc[:, 'Gphi2'] = getPhi(gam2)

    pi0 = vecAdd(gam, gam2)
    VGS = [-df_epgg['Epx'], -df_epgg['Epy'], pbeam - df_epgg['Epz']]
    v3l = cross(beam, ele)
    v3h = cross(pro, VGS)
    v3g = cross(VGS, gam)
    VmissPi0 = [-df_epgg["Epx"] - df_epgg["Ppx"], -df_epgg["Epy"] -
                df_epgg["Ppy"], pbeam - df_epgg["Epz"] - df_epgg["Ppz"]]
    VmissP = [-df_epgg["Epx"] - df_epgg["Gpx"] - df_epgg["Gpx2"], -df_epgg["Epy"] -
                df_epgg["Gpy"] - df_epgg["Gpy2"], pbeam - df_epgg["Epz"] - df_epgg["Gpz"] - df_epgg["Gpz2"]]
    Vmiss = [-df_epgg["Epx"] - df_epgg["Ppx"] - df_epgg["Gpx"] - df_epgg["Gpx2"],
                -df_epgg["Epy"] - df_epgg["Ppy"] - df_epgg["Gpy"] - df_epgg["Gpy2"],
                pbeam - df_epgg["Epz"] - df_epgg["Ppz"] - df_epgg["Gpz"] - df_epgg["Gpz2"]]

    df_epgg.loc[:, 'Mpx'], df_epgg.loc[:, 'Mpy'], df_epgg.loc[:, 'Mpz'] = Vmiss

    # binning kinematics
    df_epgg.loc[:,'Q2'] = -((ebeam - df_epgg['Ee'])**2 - mag2(VGS))
    df_epgg.loc[:,'nu'] = (ebeam - df_epgg['Ee'])
    df_epgg.loc[:,'xB'] = df_epgg['Q2'] / 2.0 / M / df_epgg['nu']
    df_epgg.loc[:,'t'] = 2 * M * (df_epgg['Pe'] - M)
    df_epgg.loc[:,'W'] = np.sqrt(np.maximum(0, (ebeam + M - df_epgg['Ee'])**2 - mag2(VGS)))
    df_epgg.loc[:,'MPt'] = np.sqrt((df_epgg["Epx"] + df_epgg["Ppx"] + df_epgg["Gpx"] + df_epgg["Gpx2"])**2 +
                                (df_epgg["Epy"] + df_epgg["Ppy"] + df_epgg["Gpy"] + df_epgg["Gpy2"])**2)

    # trento angles
    df_epgg['phi1'] = angle(v3l, v3h)
    df_epgg['phi1'] = np.where(dot(v3l, pro) > 0, 360.0 -
                                df_epgg['phi1'], df_epgg['phi1'])
    df_epgg['phi2'] = angle(v3l, v3g)
    df_epgg['phi2'] = np.where(dot(VGS, cross(v3l, v3g)) <
                                0, 360.0 - df_epgg['phi2'], df_epgg['phi2'])


    # exclusivity variables
    df_epgg.loc[:,'MM2_ep'] = (-M - ebeam + df_epgg["Ee"] +
                            df_epgg["Pe"])**2 - mag2(VmissPi0)
    df_epgg.loc[:,'MM2_egg'] = (-M - ebeam + df_epgg["Ee"] +
                            df_epgg["Ge"] + df_epgg["Ge2"])**2 - mag2(VmissP)
    df_epgg.loc[:,'MM2_epgg'] = (-M - ebeam + df_epgg["Ee"] + df_epgg["Pe"] +
                            df_epgg["Ge"] + df_epgg["Ge2"])**2 - mag2(Vmiss)
    df_epgg.loc[:,'ME_epgg'] = (M + ebeam - df_epgg["Ee"] - df_epgg["Pe"] - df_epgg["Ge"] - df_epgg["Ge2"])
    df_epgg.loc[:,'Mpi0'] = pi0InvMass(gam, gam2)
    df_epgg.loc[:,'reconPiAngleDiff'] = angle(VmissPi0, pi0)
    df_epgg.loc[:,"Pie"] = df_epgg['Ge'] + df_epgg['Ge2']
    
    df_math_epgg = df_epgg

    return df_math_epgg


def cutDVpi(df_math_epgg):
    #make dvpi0 pairs
    df_epgg = df_math_epgg

    df_epgg.loc[:, "closeness"] = np.abs(df_epgg.loc[:, "Mpi0"] - .1349766)

    cut_xBupper = df_epgg.loc[:, "xB"] < 1  # xB
    cut_xBlower = df_epgg.loc[:, "xB"] > 0  # xB
    cut_Q2 = df_epgg.loc[:, "Q2"] > 1  # Q2
    cut_W = df_epgg.loc[:, "W"] > 2  # W

    # Exclusivity cuts
    cut_mmep = df_epgg.loc[:, "MM2_ep"] < 0.7  # mmep
    cut_meepgg = df_epgg.loc[:, "ME_epgg"] < 0.7  # meepgg
    cut_mpt = df_epgg.loc[:, "MPt"] < 0.2  # mpt
    cut_recon = df_epgg.loc[:, "reconPiAngleDiff"] < 2  # recon gam angle
    cut_pi0upper = df_epgg.loc[:, "Mpi0"] < 0.2
    cut_pi0lower = df_epgg.loc[:, "Mpi0"] > 0.07
    cut_sector = (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector"]) & (df_epgg.loc[:, "Esector"]!=df_epgg.loc[:, "Gsector2"])

    df_dvpi0 = df_epgg.loc[cut_xBupper & cut_xBlower & cut_Q2 & cut_W & cut_mmep & cut_meepgg &
                        cut_mpt & cut_recon & cut_pi0upper & cut_pi0lower & cut_sector, :]

    #For an event, there can be two gg's passed conditions above.
    #Take only one gg's that makes pi0 invariant mass
    #This case is very rare.
    #For now, duplicated proton is not considered.
    df_dvpi0.sort_values(by='closeness', ascending=False)
    df_dvpi0.sort_values(by='event')        
    
    #COMMENT THIS BACK IN TO REMOVE DUPLICATES
    #df_dvpi0 = df_dvpi0.loc[~df_dvpi0.event.duplicated(), :]

    #df_x = df_dvpi0.loc[:, ["event", "Epx", "Epy", "Epz", "Ep", "Ephi", "Etheta", "Ppx", "Ppy", "Ppz", "Pp", "Pphi", "Ptheta", "Gpx", "Gpy", "Gpz", "Gp", "Gtheta", "Gphi", "Gpx2", "Gpy2", "Gpz2", "Gp2", "Gtheta2", "Gphi2"]]
    #self.df_x = df_x #done with saving x

    return df_dvpi0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get args",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-f","--fname", help="a single root file to convert into pickles", default="infile.root")
    parser.add_argument("-o","--out", help="a single pickle file name as an output", default="outfile.pkl")
    parser.add_argument("-s","--entry_stop", help="entry_stop to stop reading the root file", default = None)
    
    args = parser.parse_args()


    df_recon = pd.read_pickle("recon/df_recon_all_9628_files.pkl")

    ic(df_recon)
    # #Calculate pi0 parameters    
    df_recon_pi0vars = makeDVpi0vars(df_recon)
    #Apply exclusivity cuts    
    df_after_cuts = cutDVpi(df_recon_pi0vars)

    ic(df_after_cuts)
    df_after_cuts.to_pickle("recon_pi0_with_dups.pkl")
    sys.exit()
    # t ranges:
    # 0.2 - 0.3, .4, .6, 1.0
    # xb ranges:
    # 0.25, 0.3, 0.38
    # q2 ranges:
    # 3, 3.5, 4

    # #xxxxxxxxxxxxxxxxxxxxxxxxxxx
    # #Push through the generated data
    # #xxxxxxxxxxxxxxxxxxxxxxxxxxx
    #df_gen = pd.read_pickle("test/df_gen.pkl")
    df_gen = pd.read_pickle("gen/df_gen_only_9.pkl")
    #df_gen = pd.read_pickle("df_genONLY.pkl")
    #df_gen = pd.read_pickle("df_gen_only_1.pkl")
    #df_gen = pd.read_pickle("df_gen_all_9628_files.pkl")
    #df_recon = pd.read_pickle("df_recon.pkl")
    #df_recon = pd.read_pickle("df_recon_all_9628_files.pkl")
    
    ic(df_gen)
    ic(df_gen.columns.values)
    #Calculate pi0 parameters    
    df_gen_pi0vars = makeGenDVpi0vars(df_gen)

    df_gen_pi0vars.to_pickle("gen_9_withpi0.pkl")
    ic(df_gen_pi0vars)
    sys.exit()
    #make plots before cuts are applied
    #hist = df_gen_pi0vars.hist(bins=30)
    #plt.show()

    x_data = df_gen_pi0vars["GenxB"]
    y_data = df_gen_pi0vars["GenQ2"]
    var_names = ["xB","Q^2 (GeV$^2$)"]
    ranges = [0,1,100,0,12,120]
    output_dir = "."
    lund_q2_xb_title = "Q^2 vs xB for Generated Events"
    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                    saveplot=False,pics_dir=output_dir,plot_title=lund_q2_xb_title.replace("/",""))

   
    
    make_histos.plot_1dhist(x_data,["xB"],[0,1,100],
                    saveplot=False,pics_dir=output_dir,plot_title=lund_q2_xb_title.replace("/",""))

    x_data = df_gen_pi0vars["Genphi1"]
    y_data = df_gen_pi0vars["Gent"]
    var_names = ["phi","t"]
    ranges = [-10,370,100,0,1.2,120]
    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                    saveplot=False,pics_dir=output_dir,plot_title=lund_q2_xb_title.replace("/",""))

    df_small_gen = df_gen_pi0vars.query("GenxB>0.3 & GenxB<0.38 & GenQ2>3 & GenQ2<3.5 & Gent> 0.5 & Gent<1")
    
    y_data = df_small_gen["GenPtheta"]
    x_data = df_small_gen["GenPphi"]
    var_names = ["phi","theta"]
    ranges = [-180,180,180,0,50,50]
    output_dir = "."
    lund_q2_xb_title = "Proton angles for {}".format("here/")

    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                    saveplot=False,pics_dir=output_dir,plot_title=lund_q2_xb_title.replace("/",""))


    y_data = df_small_gen["GenEtheta"]
    x_data = df_small_gen["GenEphi"]
    ranges = [-180,180,180,10,18,50]
    lund_q2_xb_title = "Electron angles for {}".format("here/")

    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                    saveplot=False,pics_dir=output_dir,plot_title=lund_q2_xb_title.replace("/",""))

    y_data = df_small_gen["GenGtheta"]
    x_data = df_small_gen["GenGphi"]
    ranges = [-180,180,180,5,25,50]
    lund_q2_xb_title = "Photon angles for {}".format("here/")

    make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
                    saveplot=False,pics_dir=output_dir,plot_title=lund_q2_xb_title.replace("/",""))



    x_data2 = df_small_gen["Genphi1"]
    var_names = ["phi"]
    make_histos.plot_1dhist(x_data,var_names,[0,360,20],
                    saveplot=False,pics_dir=output_dir,plot_title=lund_q2_xb_title.replace("/",""))

    sys.exit()


    # #xxxxxxxxxxxxxxxxxxxxxxxxxxx
    # #Push though the recon data
    # #xxxxxxxxxxxxxxxxxxxxxxxxxxx
    
    # ic(df_recon)
    # #Calculate pi0 parameters    
    # df_recon_pi0vars = makeDVpi0vars(df_recon)
    
    # #make plots before cuts are applied
    # #hist = df_recon_pi0vars.hist(bins=30)
    # #plt.show()

    # #Apply exclusivity cuts    
    # df_after_cuts = cutDVpi(df_recon_pi0vars)

    # #make plots after cuts are applied
    # ic(df_after_cuts)
    # #hist = df_after_cuts.hist(bins=30)
    # #plt.show()



    # df_small_gen = df_after_cuts.query("xB>0.3 & xB<0.38 & Q2>3 & Q2<3.5 & t> 0.5 & t<1")
    
    # x_data = df_small_gen["phi1"]
    # var_names = ["phi"]


    # y_data = df_small_gen["Ptheta"]
    # x_data = df_small_gen["Pphi"]
    # var_names = ["phi","theta"]
    # ranges = [0,180,180,0,50,50]
    # output_dir = "."
    # lund_q2_xb_title = "Proton angles for {}".format("here/")

    # make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
    #                 saveplot=False,pics_dir=output_dir,plot_title=lund_q2_xb_title.replace("/",""))


    # y_data = df_small_gen["Etheta"]
    # x_data = df_small_gen["Ephi"]
    # ranges = [0,180,180,10,18,50]
    # lund_q2_xb_title = "Electron angles for {}".format("here/")

    # make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
    #                 saveplot=False,pics_dir=output_dir,plot_title=lund_q2_xb_title.replace("/",""))

    # y_data = df_small_gen["Gtheta"]
    # x_data = df_small_gen["Gphi"]
    # ranges = [0,180,180,5,25,50]
    # lund_q2_xb_title = "Photon angles for {}".format("here/")

    # make_histos.plot_2dhist(x_data,y_data,var_names,ranges,colorbar=True,
    #                 saveplot=False,pics_dir=output_dir,plot_title=lund_q2_xb_title.replace("/",""))



    # #make_histos.plot_1dhist(x_data2,var_names,[0,360,20],second_x=x_data,
    # #                saveplot=False,pics_dir=output_dir,plot_title=lund_q2_xb_title.replace("/",""))


    # sys.exit()

