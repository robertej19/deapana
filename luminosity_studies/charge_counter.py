#!/usr/bin/env python3
"""code is modified from Sangbaek's initial work on converting root to pandas"""

import uproot
import pandas as pd
import numpy as np
import argparse
import os, sys
from icecream import ic
import matplotlib.pyplot as plt
from copy import copy

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get args",formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-f","--fname", help="a single root file to convert into pickles", default="infile.root")
    parser.add_argument("-o","--out", help="a single pickle file name as an output", default="outfile.pkl")
    parser.add_argument("-s","--entry_stop", help="entry_stop to stop reading the root file", default = None)
    
    args = parser.parse_args()

 
    df_after_cuts = pd.read_pickle("real/F18_All_DVPi0_Events.pkl")
  
    ic(df_after_cuts.head(5))

    ic(df_after_cuts.columns.values)
    a = df_after_cuts.RunNum.unique()
    
    q_mins = []
    q_maxs = []
    run_nums = []
    
    for run_num_val in a:
        q = "RunNum == {}".format(run_num_val)
        small = df_after_cuts.query(q).beamQ
        q_mins.append(small.min())
        q_maxs.append(small.max())
        run_nums.append(run_num_val)
    
    d = {'RunNum': run_nums, 'beamQ_min': q_mins,'beamQ_max': q_maxs }
    df = pd.DataFrame(data=d)
    df['beamQ_accum'] = df['beamQ_max']-df['beamQ_min']
    total = df.beamQ_accum.sum()
    ic(df)
    ic(total)
    df.to_csv(r'beamq.txt')
    sys.exit()