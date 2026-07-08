import numpy as np


def compute(adj, vertex_normals):
    """
    Curvature estimated from neighboring normal variation.
    """

    curvature = np.zeros(len(vertex_normals))

    for vertex, neighbors in adj.items():

        if not neighbors:
            continue

        n = vertex_normals[vertex]

        total = 0.0

        for nb in neighbors:
            total += 1.0 - np.dot(
                n,
                vertex_normals[nb],
            )

        curvature[vertex] = total / len(neighbors)

    return curvature
