# This is the streamlined version of caluclating LDOS
# Written by Anderson Steckler June 2025

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import pybinding as pb
from pybinding.repository import graphene
from scipy.integrate import trapezoid

from tqdm import tqdm

import argparse
import psutil
import os

# Parser CGPT
parser = argparse.ArgumentParser(description="A simple argparse example")
parser.add_argument("sep", help="number of unit cells of seperation")
parser.add_argument("outdir", help="path to output folder")

args = parser.parse_args()

sep_num = args.sep
outdir = args.outdir

# Creating output folder
os.makedirs(outdir, exist_ok=True)

# Basis Vectors
a1 = graphene.a * np.array([1, np.sqrt(3)])/2
a2 = graphene.a * np.array([-1, np.sqrt(3)])/2
a3 = graphene.a * np.array([1, 0])

at = graphene.a/np.sqrt(3) # Distance between A and B sublattices

# Creating vacancy
def vacancy(position, radius):
    @pb.site_state_modifier
    def modifier(state, x, y):
        x0, y0 = position
        state[(x-x0)**2 + (y-y0)**2 < radius**2] = False
        return state
    return modifier

#width of hexagon
h = at/2 * 3/np.sqrt(3)
n = int(sep_num)

## For two defects horizontal
model = pb.Model(graphene.monolayer(), 
                 pb.rectangle(50, 50), 
                 vacancy(position=[(2 * n) * at/2, at/2], radius=0.09),
                 vacancy(position=[(2 * n) * -at/2, at/2], radius=0.09))

# Lattice Plot
model.plot()
plt.xlim(-7, 7)
plt.ylim(-7, 7)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.title("Defect on Lattice")

plt.savefig(outdir + "/defect_on_lattice.png")

# Discrete spatial LDOS
kpm = pb.kpm(model) # Kernal polynomial method. Instead of diagonalizing, explands everything in terms of Chebyshev polynomials

size = 20 # half of lateral size
energies = np.linspace(0, 1, 100) # eV

spatial_ldos = kpm.calc_spatial_ldos(energies,
                                     broadening=0.1, #eV
                                     shape=pb.rectangle(size*2))

# Plotting
fig2, ax2 = plt.subplots()

smap = spatial_ldos.structure_map(energies[0])
smap.plot(site_radius=(0.02, 0.15))
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)
ax2.set_xlabel("x (nm)")
ax2.set_ylabel("y (nm)")
ax2.set_title("Discrete LDOS at " + str(energies[0]) + "eV")

fig2.savefig(outdir + "/discrete_ldos.png")

# Create 2D Arr so we can use scipy integration
ldos_arr = []
for en in energies:
    ldos_arr.append(spatial_ldos.structure_map(en).data)


ldos_arr = np.array(ldos_arr) # now rows are energy and cols are LDOS values

# Integrate
sum_ldos = trapezoid(ldos_arr, x=energies, axis=0)

print(sum_ldos.shape)

# Continuous LDOS
(X, Y, Z) = smap.spatial_map.positions

# integrated version
psi2 = sum_ldos

mesh_size = 1000
x, y = np.meshgrid(np.linspace(-10, 10, mesh_size),
                   np.linspace(-10, 10, mesh_size))
z = x*0

# Interpolate by gaussian
# Every site impacts the points like a gaussian
# See Slater-Koster tightbinding for graphene band structure
for (X, Y, psi2) in tqdm(zip(X, Y, psi2)):
  z += psi2*np.exp(-((x-X)**2+(y-Y)**2)/(at)**2)

z_min, z_max = 0, np.abs(z).max()

fig3, ax3 = plt.subplots()

c = ax3.pcolormesh(x, y, np.abs(z), cmap='bwr', vmin=z_min, vmax=z_max)
ax3.axis([x.min(), x.max(), y.min(), y.max()])
fig3.colorbar(c, ax=ax3, label="U (eV)")
ax3.set_aspect("equal")
#ax3.set_xlim(-5, 5)
#ax3.set_ylim(-5, 5)

ax3.set_title("Continuous Spatial LDOS")
fig3.savefig(outdir + "/continuous_ldos.png")

df = pd.DataFrame()

df['x'] = x.flatten()
df['y'] = y.flatten()
df['Re(z)'] = np.real(z).flatten()
#df['Im(z)'] = np.imag(z).flatten()
#df['MeshSize'] = mesh_size
#df['DiscreteSize'] = size
#df['LDOS Energy'] = energies[0]

df.to_csv(outdir + "/continous_ldos.csv", index=False)

