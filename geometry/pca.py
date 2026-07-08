import numpy as np


def compute_local_frame(points, neighborhood):
    """
    Compute a local tangent frame using PCA.

    Returns
    -------
    centroid : (3,)
    basis    : (3,3)

        basis[:,0] = tangent 1
        basis[:,1] = tangent 2
        basis[:,2] = normal
    """

    pts = points[neighborhood]

    # ----------------------------
    # Centroid
    # ----------------------------

    centroid = pts.mean(axis=0)

    centered = pts - centroid

    # ----------------------------
    # Covariance matrix
    # ----------------------------

    C = centered.T @ centered

    # ----------------------------
    # Eigen decomposition
    # ----------------------------

    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Smallest eigenvalue = normal

    order = np.argsort(eigenvalues)

    basis = eigenvectors[:, order]
    mesh_normal = points[neighborhood[0]] - centroid

    if np.dot(basis[:, 2], mesh_normal) < 0:
        basis[:, 2] *= -1

    return centroid, basis