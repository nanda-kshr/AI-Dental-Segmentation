import numpy as np


def project_to_local_frame(points, neighborhood, centroid, basis):

    pts = points[neighborhood]

    centered = pts - centroid

    local = centered @ basis

    u = local[:, 0]
    v = local[:, 1]
    w = local[:, 2]

    return u, v, w


def fit_quadric(u, v, w):

    A = np.column_stack([
        u * u,
        u * v,
        v * v,
        u,
        v,
        np.ones_like(u),
    ])

    coeffs, _, _, _ = np.linalg.lstsq(
        A,
        w,
        rcond=None,
    )

    return coeffs


def principal_curvatures(coeffs):

    a, b, c, d, e, f = coeffs

    # Hessian

    H = np.array([
        [2 * a, b],
        [b, 2 * c],
    ])

    eigvals = np.linalg.eigvalsh(H)

    k1 = eigvals[1]
    k2 = eigvals[0]

    return k1, k2


def mean_curvature(k1, k2):
    return (k1 + k2) / 2.0


def gaussian_curvature(k1, k2):
    return k1 * k2


def curvedness(k1, k2):

    return np.sqrt(
        (k1 * k1 + k2 * k2) / 2.0
    )


def shape_index(k1, k2):

    if abs(k1 - k2) < 1e-10:
        return 0.0

    return (
        2.0
        / np.pi
        * np.arctan(
            (k1 + k2)
            / (k1 - k2)
        )
    )