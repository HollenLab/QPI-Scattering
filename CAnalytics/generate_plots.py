import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

a = 0.24595 #nm

fpath = r"/home/cmp/Documents/Github/QPI-Scattering/CAnalytics/ldos.tsv"
df = pd.read_csv(fpath, sep="\t", header=None, dtype=np.double, skiprows=3)

# Load in information about scale, dimensions, etc
head = pd.read_csv(fpath, sep="\t", nrows=1)

nx, ny, dx, dy, sx, sy, sep = head.iloc[0].tolist()

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

plt.show()