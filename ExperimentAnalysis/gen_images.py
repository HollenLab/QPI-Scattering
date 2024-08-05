"""
gen_images.py -- Program to generate images used in movies to see changes in how filtering affects spatial plots

Author: Anderson Steckler
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

# Degrees
def rotate2D(x, y, theta):
    theta = np.deg2rad(theta)
    a = x * np.cos(theta) - y * np.sin(theta)
    b = x * np.sin(theta) + y * np.cos(theta)

    return np.array([a, b])

# Periodic 1D Hann Function
def hann1dp(x, N, x0):
    return np.cos(np.pi * (x-x0)/N) ** 2

def hann2d(x, y, Nx, Ny, x0, y0): 
    val = hann1dp(x, Nx, x0) * hann1dp(y, Ny, y0)

    return np.where((np.abs(x-x0) <= 0.5 * Nx) & (np.abs(y-y0) <= 0.5 * Ny), val, 0)

def createImage(fname, rad, angle):
    # Import Data, create Dataframe
    #pth = r"Data/STM_Device 115_g 10K_2019_10_11_17_03_18_195 Image Topography.txt"
    pth = r"/mnt/c/Users/ander/OneDrive/Documents/GitHub/QPI-Scattering/ExperimentAnalysis/Data/STM_Device 115_g 10K_2019_10_11_17_03_18_195 Image Topography.txt"
    df = pd.read_table(pth, skiprows=3, sep='\t', header=None, names=['X', 'Y', 'Z'])

    # Turn into matrix CGPT
    ux, uy = np.unique(df['X']), np.unique(df['Y'])

    z = np.zeros((len(ux), len(uy)))

    for i, x in enumerate(ux):
        for j, y in enumerate(uy):
            index = np.where((df['X'] == x) & (df['Y'] == y))[0][0]
            z[j, i] = df['Z'][index]

    # Apply FFT
    ftz = np.fft.fft2(z)
    ftz = np.fft.fftshift(ftz)
    plz = np.log(np.abs(ftz ** 2))

    # Frequency Points
    dx = np.array([ux[i+1] - ux[i] for i in range(0, len(ux)-1)]) # Get spacing
    dy = np.array([uy[i+1] - uy[i] for i in range(0, len(uy)-1)])

    # Spacing is not exact (floating point) so we average
    # But its close enough to do this with no problem
    dxm = np.average(dx)
    dym = np.average(dy)

    fx = np.fft.fftfreq(len(ux), dxm)
    fy = np.fft.fftfreq(len(uy), dym)
    fxs = np.fft.fftshift(fx)
    fys = np.fft.fftshift(fy)

    # Back Scatter Point
    acc=0.142 #nm
    deltaKx = 4 * np.pi / (3 * np.sqrt(3) * acc) * (1/(2 * np.pi))
    dKR_x, dKR_y = rotate2D(deltaKx, 0, angle)

    # Hann Filtering
    hann_vec = np.vectorize(hann2d)

    ffx, ffy = np.meshgrid(fxs, fys)
    hann_z = hann2d(ffx, ffy, 2 * rad, 2 * rad, dKR_x, dKR_y) + hann2d(ffx, ffy, 2 * rad, 2 * rad, -dKR_x, -dKR_y) 

    ftzf = ftz * hann_z

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].pcolormesh(fxs, fys, np.abs(ftzf))
    axs[0].set_xlabel("1/x (nm^-1)")
    axs[0].set_ylabel("1/y (nm^-1)")
    axs[0].set_title(f"Hann at DK width={rad*2:.2f} angle={angle:.2f}")
    axs[0].set_aspect("equal")

    # Inverse FFT
    z_filtered = np.fft.ifft2(ftzf)

    axs[1].pcolormesh(fxs, fys, np.log(np.abs(z_filtered)))
    axs[1].set_xlabel("x (nm)")
    axs[1].set_ylabel("y (nm)")
    axs[1].set_title("STM_Device 115_g 10K_2019_10_11_17_03_18_195")
    axs[1].set_aspect("equal")

    out = r"/mnt/c/Users/ander/OneDrive/Documents/GitHub/QPI-Scattering/ExperimentAnalysis/Images/ChangeAngle/"
    plt.tight_layout()
    fig.savefig(out + fname + ".png")


frames = 60 * 3
rad_list = np.linspace(0, 360, frames) #60 FPS for 3 seconds
for n, r in tqdm(enumerate(rad_list)):
    createImage(f"radius{n:d}", 0.37/2, r)
