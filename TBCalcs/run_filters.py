import os
import sys
import glob
import subprocess

rname = "ab_horizontal"

truep = r"/mnt/c/Users/ander/OneDrive/Documents/GitHub/QPI-Scattering/TBCalcs/"
dfolder = r"Calculations/" + rname + "/"

pypath = "/mnt/c/Users/ander/OneDrive/Documents/GitHub/QPI-Scattering/TBCalcs/filterk.py"

folds = [d for d in glob.glob(truep + dfolder + '/*') if os.path.isdir(d)]
basenames = [os.path.basename(fs) for fs in folds]

for bnames in basenames:
    print(bnames)
    cmd = [sys.executable, pypath, rname, bnames]
    result = subprocess.run(cmd)