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


data_dir = "data/binned/"

parser = argparse.ArgumentParser(description="Get args",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d","--dirname", help="a directory of pickle files", default=data_dir)
parser.add_argument("-m","--merge", help="merge all pkl files", default = False, action='store_true')
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

    rfinal["acc_cor"] = rfinal["total_gen_count"]/rfinal["recon_counts"]
    ic(rfinal)
    sys.exit()


    outname = args.dirname+"merged_"+str(len(files))+".pkl"
    ic(outname)
    results.to_pickle(outname)



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

