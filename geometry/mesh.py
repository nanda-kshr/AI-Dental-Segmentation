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


def laplacian_smooth(values, adj, iterations=10, lambda_factor=0.5):
    """Smooth vertex values using Laplacian smoothing over the adjacency graph."""
    smoothed = np.array(values, dtype=float)
    # Convert adjacency set to lists once for speed
    adj_lists = {v: list(nbs) for v, nbs in adj.items()}
    
    for _ in range(iterations):
        new_smoothed = smoothed.copy()
        for vertex, neighbors in adj_lists.items():
            if neighbors:
                neighbor_avg = np.mean(smoothed[neighbors])
                new_smoothed[vertex] = (
                    (1.0 - lambda_factor) * smoothed[vertex] +
                    lambda_factor * neighbor_avg
                )
        smoothed = new_smoothed
    return smoothed