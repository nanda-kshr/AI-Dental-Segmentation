import numpy as np
import pyvista as pv

from geometry.mesh import (
    extract_faces,
    build_adjacency,
    vertex_degrees,
    laplacian_smooth,
)

from geometry.normals import (
    compute_face_geometry,
    compute_vertex_normals,
)

from geometry.curvature import normal_variation

from geometry.viewer import GeometryViewer


# ------------------------------------------------------
# Load mesh
# ------------------------------------------------------

mesh = pv.read("data/raw/HarithaLakshmi LowerJawScan.stl")      # <-- your STL

points = mesh.points

print(mesh)

print(f"Vertices : {mesh.n_points}")
print(f"Triangles: {mesh.n_cells}")

# ------------------------------------------------------
# Mesh topology
# ------------------------------------------------------

faces = extract_faces(mesh)

adj = build_adjacency(
    faces,
    mesh.n_points,
)

degrees = vertex_degrees(
    adj,
    mesh.n_points,
)

print()

print("Adjacency")
print("-------------------")
print("Average degree :", np.mean(degrees))

# ------------------------------------------------------
# Geometry
# ------------------------------------------------------

raw_face_normals, face_normals, face_areas = compute_face_geometry(
    points,
    faces,
)

vertex_normals = compute_vertex_normals(
    mesh.n_points,
    faces,
    raw_face_normals,
)

# ------------------------------------------------------
# Curvature
# ------------------------------------------------------

raw_curvature = normal_variation.compute(
    adj,
    vertex_normals,
)

# Apply Laplacian smoothing to diffuse the curvature (creating smooth weight paint gradients)
curvature = laplacian_smooth(raw_curvature, adj, iterations=10)

print()

print("Curvature (Smoothed)")
print("-------------------")
print("Min :", curvature.min())
print("Max :", curvature.max())
print("Mean:", curvature.mean())

# ------------------------------------------------------
# Viewer
# ------------------------------------------------------

viewer = GeometryViewer(mesh)
viewer.show(curvature, title="Normal Variation")