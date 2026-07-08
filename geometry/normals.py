import numpy as np


def compute_face_geometry(points, faces):
    """
    Computes:
        - raw face normals (length = 2 * triangle area)
        - unit face normals
        - triangle areas
    """

    raw_normals = np.zeros((len(faces), 3))
    unit_normals = np.zeros((len(faces), 3))
    areas = np.zeros(len(faces))

    for i, (a, b, c) in enumerate(faces):

        p0 = points[a]
        p1 = points[b]
        p2 = points[c]

        e1 = p1 - p0
        e2 = p2 - p0

        raw = np.cross(e1, e2)

        length = np.linalg.norm(raw)

        raw_normals[i] = raw
        areas[i] = length * 0.5

        if length > 1e-12:
            unit_normals[i] = raw / length

    return raw_normals, unit_normals, areas


def compute_vertex_normals(
    num_vertices,
    faces,
    raw_face_normals,
):
    """
    Area weighted vertex normals.
    """

    vertex_normals = np.zeros((num_vertices, 3))

    for face_index, (a, b, c) in enumerate(faces):

        n = raw_face_normals[face_index]

        vertex_normals[a] += n
        vertex_normals[b] += n
        vertex_normals[c] += n

    lengths = np.linalg.norm(vertex_normals, axis=1)

    valid = lengths > 1e-12

    vertex_normals[valid] /= lengths[valid][:, None]

    return vertex_normals