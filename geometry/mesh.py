from collections import defaultdict

import numpy as np


def extract_faces(mesh):
    """Return an (N,3) array of triangle indices."""
    return mesh.faces.reshape((-1, 4))[:, 1:]


def build_adjacency(faces, num_vertices):
    """Build vertex adjacency graph."""
    adj = defaultdict(set)

    for a, b, c in faces:
        adj[a].update((b, c))
        adj[b].update((a, c))
        adj[c].update((a, b))

    return adj


def vertex_degrees(adj, num_vertices):
    return [len(adj[i]) for i in range(num_vertices)]