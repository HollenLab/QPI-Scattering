import glob
import os
import re
import numpy as np
import pandas as pd

from tqdm import tqdm

print("Summing over data...")

def extractSep(fdir):
    match = re.search(r'sep(\d+)a', fdir)
    if match:
        number = match.group(1)
        print("Extracted number:", number)
        return int(number)
    else:
        print("Pattern not found.")
        return -1

def sumData(folderdir):

    sepnum = extractSep(folderdir)
    print("Summing for sep ", sepnum, "a")

    tsv_files = glob.glob(folderdir + "/*.tsv")

    # Load in information about scale, dimensions, etc
    head = pd.read_csv(tsv_files[0], sep="\t", nrows=1)

    nx, ny, dx, dy, sx, sy, sep = head.iloc[0].tolist()

    data = np.zeros((int(nx), int(ny)))

    for file in tqdm(tsv_files):
        df = pd.read_csv(file, sep="\t", header=None, dtype=np.double, skiprows=3)
        array = df.to_numpy()

        data += array

    np.savetxt(folderdir + "/sum_ldos.csv", data, delimiter=",", header=str(head.iloc[0].tolist()))

#####################################

a = 0.24595 #nm

dirfolders = glob.glob("/home/cmp/Documents/Github/QPI-Scattering/CAnalytics/output/*")

for fd in dirfolders:
    if os.path.isdir(fd):
        sumData(fd)
