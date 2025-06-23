# single filter at r
# companion to ldos.py
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import skimage

import pybinding as pb
from pybinding.repository import graphene

import os

from tqdm import tqdm
import argparse
import psutil
import os

# Parser CGPT
parser = argparse.ArgumentParser(description="A simple argparse example")
parser.add_argument("radius", help="hann radius")

args = parser.parse_args()

hannr = args.radius

fpath = r"Calculations/integrated_asub_horizontal/sep0/continous_ldos.csv"
df=pd.read_csv(fpath)

# Get the sample name to use on graphs from directory
dir_path = os.path.dirname(fpath)
sample_name = os.path.basename(dir_path)
print(sample_name)

w = int(np.sqrt(np.size(df['x'])))

x = np.reshape(df['x'], (w, w))
y = np.reshape(df['y'], (w, w))
z = np.reshape(df['Re(z)'], (w, w))
#z = np.reshape(df['Re(z)'] + 1j * df['Im(z)'], (w, w)).astype(np.complex128)

# FFT
ftz = np.fft.fft2(z)
ftz = np.fft.fftshift(ftz) # Shifts zero component freq to center

# Log scale
plz = np.log(np.abs(ftz ** 2)) # I FOUND THE SQUARE?

# Recreating linspace
#l = np.linspace(-20, 20, 1000)
#stepl = l[1] - l[0]
#fx = np.fft.fftfreq(len(l), stepl)
#fy = np.fft.fftfreq(len(l), stepl)
#fxs = np.fft.fftshift(fx)
#fys = np.fft.fftshift(fy)

ux, uy = np.unique(df['x']), np.unique(df['y'])
dx = np.array([ux[i+1] - ux[i] for i in range(0, len(ux)-1)])
dy = np.array([uy[i+1] - uy[i] for i in range(0, len(uy)-1)])

dxm = np.average(dx)
dym = np.average(dy)

# Using df['x'] and df['y'] spacing
fx = np.fft.fftfreq(len(ux), dxm)
fy = np.fft.fftfreq(len(uy), dym)
fxs = np.fft.fftshift(fx)
fys = np.fft.fftshift(fy)

fig4, ax4 = plt.subplots(figsize=(7, 7))
ax4.pcolormesh(fxs, fys, plz)

# From liu
a=0.24595 #nm
acc=0.142 #nm
deltaKx = 4 * np.pi / (3 * np.sqrt(3) * acc) * (1/(2 * np.pi))

# Rotate by 60 to get other point
# This comes from rotation matrix
dKR_x = 0.5 * deltaKx
dKR_y = np.sqrt(3)/2 * deltaKx

# Plotting delta K Point
ax4.scatter(deltaKx, 0, marker='x', color='red')

frac = (2 * np.pi)/(3 * a)
bx = np.sqrt(frac ** 2 + 3 * (frac) ** 2) # Spatial lattice point
ax4.scatter(0, np.sqrt(3) * (bx) * (1/(2 * np.pi)), marker='x', color='blue')
ax4.scatter(dKR_x, dKR_y, marker='x', color='red')

ax4.set_xlabel("x (nm)^-1")
ax4.set_ylabel("y (nm)^-1")
ax4.set_aspect("equal")

ax4.set_title("2D FFT: " + sample_name)

ax4.set_xlim(-5, 5)
ax4.set_ylim(-5, 5)

#fig4.savefig("freqout/fft.png")

# Periodic 1D Hann Function
def hann1dp(x, N, x0):
    return np.cos(np.pi * (x-x0)/N) ** 2

def hann2d(x, y, Nx, Ny, x0, y0): 
    val = hann1dp(x, Nx, x0) * hann1dp(y, Ny, y0)

    return np.where((np.abs(x-x0) <= 0.5 * Nx) & (np.abs(y-y0) <= 0.5 * Ny), val, 0)

hann_vec = np.vectorize(hann2d)

# Hann Filtering
ffx, ffy = np.meshgrid(fxs, fys)
rad = 0.4

px = np.sqrt(dKR_x ** 2 + dKR_y ** 2)
hann_z = hann2d(ffx, ffy, 2 * rad, 2 * rad, px, 0) + hann2d(ffx, ffy, 2 * rad, 2 * rad, -px, 0)

ftzf = ftz * hann_z
plzf = plz * hann_z

fig10, ax10 = plt.subplots(figsize=(8, 8))

z_filtered = np.fft.ifft2(ftzf)
zfls = np.log(np.real(z_filtered * np.conj(z_filtered)))

ux, uy = np.unique(df['x']), np.unique(df['y'])
ax10.pcolormesh(ux, uy, zfls)
ax10.set_aspect('equal')
ax10.set_title('np.log(np.real(z_filtered * np.conj(z_filtered)))')
ax10.set_xlabel('nm')
ax10.set_xlim(-5, 5)
ax10.set_ylim(-5, 5)

crad = 2.5
crc = plt.Circle((0, 0), crad, color='red', fill=False)

#ax10.add_patch(crc)
ax10.legend([f'r={crad:.2f} nm'])

m1, b1 = -dKR_x/dKR_y, 0
xlin = np.linspace(-5, 5, 100)
#ax10.plot(xlin, m1 * (xlin) + (b1), 'blue', linestyle='dashed', alpha=0.3)

wavel = 2 * np.pi / np.sqrt(dKR_x ** 2 + dKR_y ** 2) * 1/(np.pi * 2)
b2 = (m1 ** 2 + 1)/np.sqrt(m1 ** 2 + 1) * wavel
#ax10.plot(xlin, m1 * (xlin) + b2, 'blue', linestyle='dashed', alpha=0.3)

os.makedirs('fastfilter/', exist_ok=True)
fig10.savefig('fastfilter/rad'+hannr+".png")