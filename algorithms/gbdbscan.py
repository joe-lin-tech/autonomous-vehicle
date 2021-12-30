import numpy as np
import matplotlib.pyplot as plt

# TODO fix GBDBSCAN

class GBDBSCAN:

    def __init__(self, g, f, k, num_r, max_r, num_t, r_init=0):
        self.g = g
        self.f = f
        self.k = k
        self.num_r = num_r
        self.num_t = num_t
        self.labels_ = None
        self.c = np.zeros((num_r, num_t))
        for r in range(num_r):
            for t in range(num_t):
                dr = max_r / num_r
                dt = np.pi / num_t
                crt = ((r_init + dr * r) / (2.0 * dr)) * (np.sin(dt *
                                                                 (t + 1.0) - dt * t) + np.sin(dt * t - dt * (t - 1.0)))
                self.c[r][t] = crt

    def inEllipse(self, p, q, r_eps, t_eps):
        return ((q[0] - p[0]) ** 2 / r_eps ** 2 + (q[1] - p[1]) ** 2 / t_eps ** 2) <= 1

    def possibleObservations(self, p, r_eps, t_eps):

        obs = 0
        ceil_r = int(np.ceil(r_eps))
        ceil_t = int(np.ceil(t_eps))
        r_min, r_max = max(
            0, p[0] - ceil_r), min(self.num_r, p[0] + ceil_r + 1)
        t_min, t_max = max(
            0, p[1] - ceil_t), min(self.num_t, p[1] + ceil_t + 1)
        for r in range(r_min, r_max):
            for t in range(t_min, t_max):
                q = (r, t)
                if self.inEllipse(q, p, r_eps, t_eps):
                    obs += 1
        return obs

    def regionQuery(self, X, p):
        neighbors = []
        r, t = int(X[p][0] / self.num_r), int(X[p][1] / self.num_r)
        r_eps = self.g
        t_eps = self.g / (self.f * self.c[r_eps][int(X[p][1] / self.num_t)])
        for q in range(X.shape[0]):
            if self.inEllipse(X[p], X[q], r_eps, t_eps):
                neighbors.append(q)
        min_pts = self.possibleObservations((r, t), r_eps, t_eps) * self.k
        return neighbors, min_pts

    def expandCluster(self, X, point, cluster):
        neighbors, min_pts = self.regionQuery(X, point)
        if len(neighbors) < min_pts:
            self.labels_[point] = -1
            return False
        else:
            self.labels_[point] = cluster
            for n in neighbors:
                self.labels_[n] = cluster

            while len(neighbors) > 0:
                curr = neighbors[0]
                results, min_pts = self.regionQuery(X, curr)
                if len(results) >= min_pts:
                    for i in range(0, len(results)):
                        res = results[i]
                        if self.labels_[res] == -2 or self.labels_[res] == -1:
                            if self.labels_[res] == -2:
                                neighbors.append(res)
                            self.labels_[res] = cluster
                neighbors = neighbors[1:]
            return True

    def fit(self, X):
        self.labels_ = [-2] * X.shape[0]
        cluster = 1
        for point in range(X.shape[0]):
            if self.labels_[point] == -2:
                if self.expandCluster(X, point, cluster):
                    cluster += 1
        return self
