import numpy as np
import gpytoolbox as gpy
import matplotlib.pyplot as plt
import polyscope as ps
import sys

def create_hemisphere_mesh(resolution=50):
    """
    Create a hemisphere mesh using marching cubes with specified resolution.
    
    Parameters:
    -----------
    resolution : int
        Controls the grid resolution for marching cubes (higher = more vertices)
    visualize : bool
        Whether to plot the resulting mesh
    
    Returns:
    --------
    V : (n,3) numpy array of vertex positions
    F : (m,3) numpy array of triangle indices
    """
    # Create a grid for the marching cubes algorithm
    grid_size = resolution
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    z = np.linspace(0, 1, grid_size)  # Only positive z for hemisphere
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create a signed distance function for a hemisphere (radius=1)
    # SDF = sqrt(x² + y² + z²) - 1, but only for z ≥ 0
    SDF = np.sqrt(X**2 + Y**2 + Z**2) - 1.0
    SDF = SDF.flatten()

    GV = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    # Run marching cubes
    V, F = gpy.marching_cubes(SDF, GV, X.shape[0], Y.shape[0], Z.shape[0])
    
    # Keep only the upper half (z ≥ 0)
    # Marching cubes may create some vertices slightly below z=0, so we'll clean those
    V = V[V[:,2] >= -0.00]  # Small tolerance
    
    return V, F

def create_ellipsoid_mesh(resolution=50):
    # Create a grid for the marching cubes algorithm
    grid_size = resolution
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    z = np.linspace(-5, 5, grid_size) # Only positive z for hemisphere
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    SDF = (X**2/4. + Y**2/9. + Z**2/1.) - 1.0
    SDF = SDF.flatten()

    GV = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    # Run marching cubes
    V, F = gpy.marching_cubes(SDF, GV, X.shape[0], Y.shape[0], Z.shape[0])
    
    return V, F

def create_heightfield(resolution=50):
    # Create a grid for the marching cubes algorithm
    grid_size = resolution
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    z = np.linspace(-1, 1, grid_size) # Only positive z for hemisphere
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    SDF = 0.5*(X**2 + Y**2) - Z
    SDF = SDF.flatten()

    GV = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    # Run marching cubes
    V, F = gpy.marching_cubes(SDF, GV, X.shape[0], Y.shape[0], Z.shape[0])
    
    return V, F

# Example usage with different resolutions
mesh = sys.argv[1]
res = int(sys.argv[2])
if mesh == "hemisphere":
    V, F = create_hemisphere_mesh(resolution=res)
    V, F = gpy.remesh_botsch(V, F)
elif mesh == "ellipsoid":
    V, F = create_ellipsoid_mesh(resolution=res)
    V, F = gpy.remesh_botsch(V, F)
elif mesh == "heightfield":
    V, F = create_heightfield(resolution=res)
    V, F = gpy.remesh_botsch(V, F)
else:
    print("Usage: python create_mesh.py [hemisphere|ellipsoid|heightfield] [resolution]")
    exit()

print(f"Resolution {res}: {len(V)} vertices, {len(F)} faces")
gpy.write_mesh(f"{sys.argv[1]}_{len(V)}x{len(F)}.obj", V, F)