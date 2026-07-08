from scipy.spatial import cKDTree


class NeighborhoodSearch:

    def __init__(self, points):
        self.points = points
        self.tree = cKDTree(points)

    def radius(self, vertex, radius):
        center = self.points[vertex]
        return self.tree.query_ball_point(center, radius)