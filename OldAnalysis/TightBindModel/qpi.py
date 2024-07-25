import numpy as np
import matplotlib.pyplot as plt

import pybinding as pb
from pybinding.repository import graphene
from tqdm import tqdm

# Basis Vectors
a1 = graphene.a * np.array([1, np.sqrt(3)])/2
a2 = graphene.a * np.array([-1, np.sqrt(3)])/2
a3 = graphene.a * np.array([1, 0])

at = graphene.a/np.sqrt(3) # Distance between A and B sublattices

# Here we model the defect as a gaussian potential
# We can specify whether its at sublattice points or inbetween
# Duterix 2019 Defect is on sublattice A

def pv(x, y):
  return -3*np.exp(-((1)*x**2+(1)*(y-at)**2)/(at)**2) + -3*np.exp(-((1)*x**2+(1)*(y+2*at)**2)/(at)**2)

def defect():
  @pb.onsite_energy_modifier
  def potential(x, y):
    #return -3*np.exp(-(x**2+(y-at/2)**2)/(at)**2)
    return pv(x, y)

  return potential

# Create tightbinding model
model = pb.Model(graphene.monolayer(), pb.rectangle(50, 50), defect())

kpm = pb.kpm(model)

size = 20 # half of lateral size
energies = np.array([0.2]) # eV

spatial_ldos = kpm.calc_spatial_ldos(np.linspace(-3, 3, 100),
                                     broadening=0.1, #eV
                                     shape=pb.rectangle(size*2))

fig2, ax2 = plt.subplots()

smap = spatial_ldos.structure_map(energies[0])
smap.plot(site_radius=(0.02, 0.15))
ax2.set_xlim(-size, size)
ax2.set_ylim(-size, size)

print("done")

plt.show()