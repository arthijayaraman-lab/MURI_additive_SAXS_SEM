import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Generate random points within a 3D cube
cube_size = 1
num_points = 50  # Adjust for more or fewer points
points = np.random.rand(num_points, 3) * cube_size  # Random points within a unit cube

# Step 2: Apply the 3D Voronoi tessellation
vor = Voronoi(points)

# Step 3: Visualization (optional and limited)
# Plot the initial points and Voronoi vertices
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points that generated the Voronoi diagram
ax.scatter(points[:,0], points[:,1], points[:,2], color='red', label='Input Points')

# Plot the Voronoi vertices (where the Voronoi regions meet)
ax.scatter(vor.vertices[:,0], vor.vertices[:,1], vor.vertices[:,2], color='blue', label='Voronoi Vertices')

# Add labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
