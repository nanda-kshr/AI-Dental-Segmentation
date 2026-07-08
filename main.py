import pyvista as pv
import numpy as np
import time

from geometry.mesh import extract_faces, build_adjacency
from geometry.normals import compute_face_geometry, compute_vertex_normals
from geometry.curvature.normal_variation import compute as compute_normal_variation

model = "data/raw/HarithaLakshmi LowerJawScan.stl"

print("Loading mesh...")
mesh = pv.read(model)
points = mesh.points
faces = extract_faces(mesh)

print(f"Mesh loaded: {len(points)} vertices, {len(faces)} faces")

print("\nComputing Normals...")
start = time.time()
raw_face_normals, unit_face_normals, areas = compute_face_geometry(points, faces)
vertex_normals = compute_vertex_normals(len(points), faces, raw_face_normals)
elapsed = time.time() - start
print(f"✓ Face & Vertex Normals computed in {elapsed:.2f}s")

print("\nComputing Normal Variation (Curvature)...")
start = time.time()
adj = build_adjacency(faces, len(points))
normal_variation = compute_normal_variation(adj, vertex_normals)
elapsed = time.time() - start
print(f"✓ Normal variation computed in {elapsed:.2f}s")

print(f"\nNormal Variation stats:")
print(f"  Min:    {normal_variation.min():.6f}")
print(f"  Max:    {normal_variation.max():.6f}")
print(f"  Mean:   {normal_variation.mean():.6f}")
print(f"  Median: {np.median(normal_variation):.6f}")