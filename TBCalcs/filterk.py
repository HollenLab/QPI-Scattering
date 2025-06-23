import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import sys
import argparse

plt.rcParams['figure.figsize'] = [10, 10]

# Parser CGPT
parser = argparse.ArgumentParser(description="A simple argparse example")
parser.add_argument("run_set", help="The run set")
parser.add_argument("run_item", help="The item of the run set")

args = parser.parse_args()

# Path Specifics
#run_set = "ab_horizontal"
#run_item = "ab_10x10_sep=17"

run_set = args.run_set
run_item = args.run_item
dfile = "continous_ldos.csv"

# Load in data
truep = r"/home/cmp/Documents/Github/QPI-Scattering/TBCalcs/"
fpath = truep + r"Calculations/"+run_set+"/"+run_item+"/"+dfile
df=pd.read_csv(fpath)

# Specify out path
opath = truep + r"Filtering/"+run_set

if not os.path.exists(opath):
    print("Creating dir...")
    os.makedirs(opath)

opath1 = opath + '/axis1'
opath2 = opath + '/axis2'
opath3 = opath + '/axis3'
opath4 = opath + '/unfiltered'

for p in [opath1, opath2, opath3, opath4]:
    if not os.path.exists(p):
        os.makedirs(p)

# Reshaping
w = int(np.sqrt(np.size(df['x'])))

x = np.reshape(df['x'], (w, w))
y = np.reshape(df['y'], (w, w))
z = np.reshape(df['Re(z)'], (w, w))

# They should be evenly spaced, but in case there is some error we take an average
ux, uy = np.unique(df['x']), np.unique(df['y'])
dxm = np.average(np.array([ux[i+1] - ux[i] for i in range(0, len(ux)-1)]))
dym = np.average(np.array([uy[i+1] - uy[i] for i in range(0, len(uy)-1)]))

# FFT
ftz = np.fft.fft2(z)
ftz = np.fft.fftshift(ftz)
plz = np.log(np.abs(ftz ** 2))

# Creating freqs
# Using df['x'] and df['y'] spacing
fx = np.fft.fftfreq(len(ux), dxm)
fy = np.fft.fftfreq(len(uy), dym)
fxs = np.fft.fftshift(fx) * 2 * np.pi # Not sure why scale here
fys = np.fft.fftshift(fy) * 2 * np.pi

# From liu
a=0.24595 #nm
acc=0.142 #nm
deltaKx = 4 * np.pi / (3 * np.sqrt(3) * acc)

# Axes
dKx1, dKy1 = deltaKx, 0 #axis 1
dKx2, dKy2 = 0.5 * deltaKx, np.sqrt(3)/2 * deltaKx #axis 2 (60 degree rotation)
dKx3, dKy3 = -dKx2, dKy2 #axis 3

fig, ax = plt.subplots()
ax.pcolormesh(fxs, fys, plz)
ax.set_aspect('equal')
ax.plot(dKx1, dKy1, marker='x', alpha=0.9)
ax.plot(dKx2, dKy2, marker='x', alpha=0.9)
ax.plot(dKx3, dKy3, marker='x', alpha=0.9)
ax.set_xlabel('x (nm^-1)')
ax.set_ylabel('y (nm^-1)')
ax.set_xlim(-40, 40)
ax.set_ylim(-40, 40)

# Saving Unfiltered
fig.savefig(opath4 + '/kspace'+ run_item + '.png')

### FILTERING ###
# Periodic 1D Hann Function
def hann1dp(x, N, x0):
    return np.cos(np.pi * (x-x0)/N) ** 2

def hann2d(x, y, Nx, Ny, x0, y0): 
    val = hann1dp(x, Nx, x0) * hann1dp(y, Ny, y0)

    return np.where((np.abs(x-x0) <= 0.5 * Nx) & (np.abs(y-y0) <= 0.5 * Ny), val, 0)

hann_vec = np.vectorize(hann2d)

for n, (adir, xpos, ypos) in enumerate(zip([opath1, opath2, opath3], [dKx1, dKx2, dKx3], [dKy1, dKy2, dKy3])):

    kpath = adir + "/kfilter"
    rpath = adir + "/rfilter"

    if not os.path.exists(kpath):
        print("Creating dir..." + kpath)
        os.makedirs(kpath)
    
    if not os.path.exists(rpath):
        print("Creating dir..." + rpath)
        os.makedirs(rpath)

    # Hann Filtering
    ffx, ffy = np.meshgrid(fxs, fys)
    rad = 0.4 * 2 * np.pi

    # Axis 1
    hannz1 = hann2d(ffx, ffy, 2 * rad, 2 * rad, xpos, ypos) + hann2d(ffx, ffy, 2 * rad, 2 * rad, -xpos, -ypos)
    ftzf, plzf = ftz * hannz1, plz * hannz1

    figh1, axh1 = plt.subplots()
    axh1.pcolormesh(fxs, fys, plzf)
    axh1.set_aspect('equal')
    axh1.set_xlim(-20, 20)
    axh1.set_ylim(-20, 20)
    axh1.set_xlabel('x (nm^-1)')
    axh1.set_ylabel('y (nm^-1)')
    axh1.set_title('Applied Filter (log scale)')

    figh1.savefig(kpath + "/" + run_item + "-kfilter" + str(n+1) + ".png")

    # Back to rspace
    z_filtered = np.fft.ifft2(ftzf)
    zfls = np.log(np.real(z_filtered * np.conj(z_filtered)))

    figh2, axh2 = plt.subplots()
    axh2.pcolormesh(ux, uy, zfls)
    axh2.set_aspect('equal')
    axh2.set_xlabel('x (nm)')
    axh2.set_ylabel('y (nm)')

    figh2.savefig(rpath + "/" + run_item + "-rfilter" + str(n+1) + ".png")

print('done')