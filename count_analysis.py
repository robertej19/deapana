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
    sys.exit()
    dfs = []

    for f in files:
        df = pd.read_pickle(args.dirname+f)
        ic(df)
        dfs.append(df)
        
    result = pd.concat(dfs)

    ic(result)

    outname = args.dirname+"merged_"+str(len(files))+".pkl"
    ic(outname)
    result.to_pickle(outname)



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

