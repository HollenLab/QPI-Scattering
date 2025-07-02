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

    a = 0.24595 #nm

    fname = folderdir + "/sum_ldos.csv"

    df = pd.read_csv(fname, skiprows=1, header=None)

    fline = ""
    with open(fname, "r") as f:
        fline = f.readline().lstrip("#").strip()

    lst = ast.literal_eval(fline)
    arr = np.array(lst).astype(np.float128)

    nx, ny, dx, dy, sx, sy, sep = arr

    xlin = np.array([ix * dx - sx for ix in range(0, int(nx))])
    ylin = np.array([iy * dy - sy for iy in range(0, int(ny))])

    X, Y = np.meshgrid(xlin, ylin)

    # We create a mask because our functions blow up at r -> zero
    maskr = 0.25
    mask1 = ((X+(sep * a)) ** 2 + Y ** 2) <= maskr**2
    mask2 = ((X-(sep * a)) ** 2 + Y ** 2) <= maskr**2

    mask = mask1 | mask2

    Z_masked = df.values
    Z_masked[mask] = 1e-9

    fig, ax = plt.subplots(figsize=(8, 8))

    pc = ax.pcolormesh(xlin, ylin, Z_masked, cmap='seismic')
    fig.colorbar(pc)

    ax.set_aspect('equal')
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')

    fig.savefig(folderdir + "/0outplotSE.png")
    fig.savefig("/home/cmp/Documents/Github/QPI-Scattering/CAnalytics/output/plots/sep" + str(sepnum)  + "a-outplotSE.png")

    plt.close()

    #plt.show()

############################
dirfolders = glob.glob("/home/cmp/Documents/Github/QPI-Scattering/CAnalytics/output/*")

os.makedirs("/home/cmp/Documents/Github/QPI-Scattering/CAnalytics/output/plots/", exist_ok=True)
for fd in dirfolders:
    if os.path.isdir(fd):
        createPlot(fd)


print("Complete :)")