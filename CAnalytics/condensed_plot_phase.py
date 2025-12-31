import numpy as np
import pandas as pd
import ast
import re
import glob
import os
import matplotlib.pyplot as plt

def extractSep(fdir):
    match = re.search(r"r=([0-9.+-eE]+)phi=\s*([0-9.+-eE]+)", fdir)
    if match:
        crcr = float(match.group(1))
        crcphi = float(match.group(2))
        print("r =", crcr)
        print("phi =", crcphi)

        return crcphi
    else:
        print("Pattern not found.")
        return -1

def createPlot(folderdir):
    sepnum = extractSep(folderdir)

    if (sepnum == -1):
        return -1

    fname = folderdir + "/condensed-ldos.tsv"

    # Conditions
    fname_c1 = folderdir + "/cond1-ldos.tsv"
    fname_c2 = folderdir + "/cond2-ldos.tsv"

    head = pd.read_csv(fname, sep="\t", nrows=1)
    nx, ny, dx, dy, sx, sy, sep, crcR, crcPhi, sKx, sKy = head.iloc[0].tolist()

    print("Creating plot for r=", crcR, "and phi=", crcPhi)

    df = pd.read_csv(fname, sep="\t", header=None, dtype=np.double, skiprows=3)
    Z = df.to_numpy()

    dfc1 = pd.read_csv(fname_c1, sep="\t", header=None, dtype=np.double, skiprows=3)
    dfc2 = pd.read_csv(fname_c2, sep="\t", header=None, dtype=np.double, skiprows=3)

    Zc1 = np.abs(np.tanh(1e4 * dfc1.to_numpy()))
    Zc2 = np.abs(np.tanh(1e3 * dfc2.to_numpy()))

    a = 0.24595 #nm
    acc = 0.142
    h = acc * np.tan(np.pi/3)
    sep = int(sep)


    xlin = np.linspace(-sx, sx, int(nx))
    ylin = np.linspace(-sy, sy, int(ny))
    X, Y = np.meshgrid(xlin, ylin)
    
    xa = sep * h
    ya = 0

    tr_x = crcR * np.cos(crcPhi);
    tr_y = crcR * np.sin(crcPhi)

    # We create a mask because our functions blow up at r -> zero
    maskr = 0.4
    mask1 = ((X+tr_x) ** 2 + (Y+tr_y) ** 2) <= maskr**2
    mask2 = ((X-tr_x) ** 2 + (Y-tr_y) ** 2) <= maskr**2

    mask = mask1 | mask2

    fig, ax = plt.subplots(figsize=(10, 10))

    #print(np.min(Z))

    #pc = ax.pcolormesh(xlin, ylin, ZR2, cmap="winter", vmax=15, vmin=-15)
    #pc = ax.pcolormesh(xlin, ylin, Z_masked, cmap="winter", vmax=1.5)
    pc = ax.pcolormesh(xlin, ylin, Z, cmap="twilight")
    ax.scatter([tr_x, -tr_x], [tr_y, -tr_y], marker="x", color="m")
    #fig.colorbar(pc)

    plt.tight_layout()

    ax.set_title("Analytical LDOS")

    ax.set_aspect('equal')
    ax.set_xlabel('x (nm)')
    ax.set_ylabel('y (nm)')

    fig.savefig(folderdir + "/0outplotSE.png")
    fig.savefig("/home/cmp/Documents/Github/QPI-Scattering/CAnalytics/output/plots/r="+ str(crcR) + "phi=" + str(crcPhi) + "-outplot.png")

    fig1, ax1 = plt.subplots(figsize=(7, 7))

    ax1.set_title(f"r={crcR:.4f} nm\n phi={crcPhi:.4f}")

    ax1.scatter([tr_x, -tr_x], [tr_y, -tr_y], marker="x", color="m")
    ax1.pcolormesh(X, Y, Zc1, cmap="Reds_r", alpha=(1-Zc1))
    ax1.pcolormesh(X, Y, Zc2, cmap="Blues_r", alpha=(1-Zc2))
    #ax1.pcolormesh(X, Y, np.abs(Zc2), cmap="Greys_r", alpha=0.5)
    ax1.set_aspect("equal")
    fig1.savefig(folderdir + "/0cond.png")
    fig1.savefig("/home/cmp/Documents/Github/QPI-Scattering/CAnalytics/output/condplots/r="+ str(crcR) + "phi=" + str(crcPhi) + "-cond.png")

    plt.close()

    #plt.show()

############################
dirfolders = glob.glob("/home/cmp/Documents/Github/QPI-Scattering/CAnalytics/output/*")

os.makedirs("/home/cmp/Documents/Github/QPI-Scattering/CAnalytics/output/plots/", exist_ok=True)
os.makedirs("/home/cmp/Documents/Github/QPI-Scattering/CAnalytics/output/condplots/", exist_ok=True)

for fd in dirfolders:
    if os.path.isdir(fd):
        createPlot(fd)


print("Complete :)")