import os
import sys
import glob
import subprocess

rname = "a_sublattice"

truep = r"/home/cmp/Documents/Github/QPI-Scattering/TBCalcs/"
dfolder = r"Calculations/" + rname + "/"

pypath = "/home/cmp/Documents/Github/QPI-Scattering/TBCalcs/filterk.py"

folds = [d for d in glob.glob(truep + dfolder + '/*') if os.path.isdir(d)]
basenames = [os.path.basename(fs) for fs in folds]

for bnames in basenames:
    print(bnames)
    cmd = [sys.executable, pypath, rname, bnames]
    result = subprocess.run(cmd)