import numpy as np


def cotangent(u, v):
    """
    Compute cot(angle) between vectors u and v.
    """

    cross = np.linalg.norm(np.cross(u, v))

    if cross < 1e-12:
        return 0.0

    return np.dot(u, v) / cross


def compute_triangle_cotangents(points, faces):
    """
    Returns

        cotangents.shape == (num_faces, 3)

    columns:

        [:,0] -> angle at vertex A
        [:,1] -> angle at vertex B
        [:,2] -> angle at vertex C
    """

    cotangents = np.zeros((len(faces), 3))

    for i, (a, b, c) in enumerate(faces):

        A = points[a]
        B = points[b]
        C = points[c]

        # angle at A
        cotA = cotangent(B - A, C - A)

        # angle at B
        cotB = cotangent(A - B, C - B)

        # angle at C
        cotC = cotangent(A - C, B - C)

        cotangents[i] = [cotA, cotB, cotC]

    return cotangents


from collections import defaultdict


def build_cotangent_weights(faces, cotangents):
    """
    Returns

        weights[(i, j)] = cotangent weight

    where i < j
    """

    weights = defaultdict(float)

    for (a, b, c), (cotA, cotB, cotC) in zip(faces, cotangents):

        # edge BC -> opposite A
        edge = tuple(sorted((b, c)))
        weights[edge] += 0.5 * cotA

        # edge AC -> opposite B
        edge = tuple(sorted((a, c)))
        weights[edge] += 0.5 * cotB

        # edge AB -> opposite C
        edge = tuple(sorted((a, b)))
        weights[edge] += 0.5 * cotC

    return weights


from scipy.sparse import coo_matrix


def build_laplacian(num_vertices, weights):
    """
    Build the symmetric cotangent Laplacian.
    """

    rows = []
    cols = []
    data = []

    diagonal = np.zeros(num_vertices)

    for (i, j), w in weights.items():

        # off-diagonal
        rows.append(i)
        cols.append(j)
        data.append(-w)

        rows.append(j)
        cols.append(i)
        data.append(-w)

        diagonal[i] += w
        diagonal[j] += w

    # diagonal entries
    for i in range(num_vertices):
        rows.append(i)
        cols.append(i)
        data.append(diagonal[i])

    L = coo_matrix(
        (data, (rows, cols)),
        shape=(num_vertices, num_vertices),
    )

    return L.tocsr()

def compute_mean_curvature(L, points):
    """
    Compute mean curvature vector and its magnitude.
    """

    H = L @ points

    magnitude = np.linalg.norm(H, axis=1)

    return H, magnitude