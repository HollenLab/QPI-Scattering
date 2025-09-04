import numpy as np
import pandas as pd
import ast
import re
import glob
import os
import matplotlib.pyplot as plt

def extractSep(fdir):
    match = re.search(r'sep(\d+)a', fdir)
    if match:
        number = match.group(1)
        print("Extracted number:", number)
        return int(number)
    else:
        print("Pattern not found.")
        return -1

def createPlot(folderdir):
    sepnum = extractSep(folderdir)

    if (sepnum == -1):
        return -1

    print("Creating plot for sep", sepnum, "a")

    fname = folderdir + "/condensed-ldos.tsv"

    head = pd.read_csv(fname, sep="\t", nrows=1)
    nx, ny, dx, dy, sx, sy, sep = head.iloc[0].tolist()

    df = pd.read_csv(fname, sep="\t", header=None, dtype=np.double, skiprows=3)
    Z = df.to_numpy()

    a = 0.24595 #nm
    acc = 0.142
    h = acc * np.tan(np.pi/3)
    sep = int(sep)


    xlin = np.linspace(-sx, sx, int(nx))
    ylin = np.linspace(-sy, sy, int(ny))
    X, Y = np.meshgrid(xlin, ylin)
    
    xa = sep * h
    ya = 0

    # We create a mask because our functions blow up at r -> zero
    maskr = 0.25
    mask1 = ((X+xa) ** 2 + (Y+ya) ** 2) <= maskr**2
    mask2 = ((X-xa) ** 2 + (Y-ya) ** 2) <= maskr**2

    mask = mask1 | mask2

    fig, ax = plt.subplots(figsize=(10, 10))

    #Z[mask]=0

    ZR = Z * (((X+xa) ** 2 + (Y+ya) ** 2)+ ((X-xa) ** 2 + (Y-ya) ** 2))


    #pc = ax.pcolormesh(xlin, ylin, ZR2, cmap="winter", vmax=15, vmin=-15)
    #pc = ax.pcolormesh(xlin, ylin, Z_masked, cmap="winter", vmax=1.5)
    #pc = ax.pcolormesh(xlin, ylin, np.log(np.abs(ZR + 1e-9)))
    pc = ax.pcolormesh(xlin, ylin, np.log((Z + 1e-9)))
    #fig.colorbar(pc)

    plt.tight_layout()

    ax.set_title("Analytical LDOS")

    ax.set_aspect('equal')
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')

    fig.savefig(folderdir + "/0outplotSE.png")
    fig.savefig("/home/cmp/Documents/Github/QPI-Scattering/CAnalytics/output/plots/sep" + str(sepnum)  + "a-outplot.png")

    plt.close()

    #plt.show()

############################
dirfolders = glob.glob("/home/cmp/Documents/Github/QPI-Scattering/CAnalytics/output/*")

os.makedirs("/home/cmp/Documents/Github/QPI-Scattering/CAnalytics/output/plots/", exist_ok=True)
for fd in dirfolders:
    if os.path.isdir(fd):
        createPlot(fd)


print("Complete :)")