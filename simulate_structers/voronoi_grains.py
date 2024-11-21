#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:00:16 2024

@author: skronen
"""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scipy.spatial import cKDTree

import sys
import torch
from torch import nn
from torch.nn import functional as F


#from scattering_code_for_share import scatter

# Generate some random 3D points within the unit cube
points = np.random.rand(55, 3)

# Compute the 3D Voronoi diagram
vor = Voronoi(points)
#%%

#"""
# Example query point
init_cube_dim = 500
x = np.linspace(0, 1, init_cube_dim)
y = np.linspace(0, 1, init_cube_dim)
z = np.linspace(0, 1, init_cube_dim)
X, Y, Z = np.meshgrid(x, y, z)

query_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

del X, Y, Z, x, y, z

#"""

#query_points = np.random.rand(100000, 3)

# Step 1: Find the closest point (Voronoi site) to the query point
tree = cKDTree(points)
closest_point_index = tree.query(query_points)[1]

#"""
target_size = 1000

cube_recon = np.expand_dims(np.expand_dims(closest_point_index.reshape(init_cube_dim,init_cube_dim,init_cube_dim).astype(np.float32), axis=0), axis=0)

cube_recon_torch = torch.from_numpy(cube_recon)
print(cube_recon_torch.shape)
cube_interpolated = F.interpolate(cube_recon_torch, size=[target_size]*3, mode="trilinear")  # trilinear
closest_point_index = cube_interpolated.numpy().flatten()
print(closest_point_index.shape)

del cube_interpolated, cube_recon_torch, query_points, tree

#colors = plt.cm.viridis(closest_point_index/np.max(closest_point_index))
closest_point_index = np.round(closest_point_index).astype(np.uint8)
cube_index = closest_point_index.reshape([target_size]*3)

print(np.unique(cube_index).shape)
#print(np.unique(cube_index))

#"""

"""
colors = plt.cm.viridis(closest_point_index/np.max(closest_point_index))

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Set plot limits to the unit cube
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

x = np.linspace(0, 1, target_size)
y = np.linspace(0, 1, target_size)
z = np.linspace(0, 1, target_size)
X, Y, Z = np.meshgrid(x, y, z)
query_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

del X, Y, Z, x, y, z

ax.scatter(query_points.T[0], query_points.T[1], query_points.T[2], c = colors, s= 5)
plt.show()
#"""