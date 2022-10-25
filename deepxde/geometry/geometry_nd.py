from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
from scipy import stats
from sklearn import preprocessing

from .geometry import Geometry
from .sampler import sample


class Hypercube(Geometry):
    def __init__(self, xmin, xmax):
        if len(xmin) != len(xmax):
            raise ValueError("Dimensions of xmin and xmax do not match.")
        if np.any(np.array(xmin) >= np.array(xmax)):
            raise ValueError("xmin >= xmax")

        self.xmin, self.xmax = np.array(xmin), np.array(xmax)
        self.side_length = self.xmax - self.xmin
        super(Hypercube, self).__init__(
            len(xmin), (self.xmin, self.xmax), np.linalg.norm(self.side_length)
        )
        self.volume = np.prod(self.side_length)

    def inside(self, x):
        return np.logical_and(
            np.all(x >= self.xmin, axis=-1), np.all(x <= self.xmax, axis=-1)
        )

    def on_boundary(self, x):
        _on_boundary = np.logical_or(
            np.any(np.isclose(x, self.xmin), axis=-1),
            np.any(np.isclose(x, self.xmax), axis=-1),
        )
        return np.logical_and(self.inside(x), _on_boundary)

    def boundary_normal(self, x):
        _n = np.isclose(x, self.xmin) * -1.0 + np.isclose(x, self.xmax) * 1.0
        if np.any(np.count_nonzero(_n, axis=-1) > 1):
            raise ValueError(
                "{}: Method `boundary_normal` do not accept points on the vertexes.".format(
                    self.__class__.__name__
                )
            )
        return _n

    def uniform_points(self, n, boundary=True):
        dx = (self.volume / n) ** (1 / self.dim)
        xi = []
        for i in range(self.dim):
            ni = int(np.ceil(self.side_length[i] / dx))
            if boundary:
                xi.append(np.linspace(self.xmin[i], self.xmax[i], num=ni))
            else:
                xi.append(
                    np.linspace(self.xmin[i], self.xmax[i], num=ni + 1, endpoint=False)[
                        1:
                    ]
                )
        x = np.array(list(itertools.product(*xi)))
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def random_points(self, n, random="pseudo"):
        x = sample(n, self.dim, random)
        return (self.xmax - self.xmin) * x + self.xmin

    def periodic_point(self, x, component):
        y = np.copy(x)
        _on_xmin = np.isclose(y[:, component], self.xmin[component])
        _on_xmax = np.isclose(y[:, component], self.xmax[component])
        y[:, component][_on_xmin] = self.xmax[component]
        y[:, component][_on_xmax] = self.xmin[component]
        return y


class Hypersphere(Geometry):
    def __init__(self, center, radius):
        self.center, self.radius = np.array(center), radius
        super(Hypersphere, self).__init__(
            len(center), (self.center - radius, self.center + radius), 2 * radius
        )

        self._r2 = radius ** 2

    def inside(self, x):
        return np.linalg.norm(x - self.center, axis=-1) <= self.radius

    def on_boundary(self, x):
        return np.isclose(np.linalg.norm(x - self.center, axis=-1), self.radius)

    def distance2boundary_unitdirn(self, x, dirn):
        """https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
        """
        xc = x - self.center
        ad = np.dot(xc, dirn)
        return -ad + (ad ** 2 - np.sum(xc * xc, axis=-1) + self._r2) ** 0.5

    def distance2boundary(self, x, dirn):
        return self.distance2boundary_unitdirn(x, dirn / np.linalg.norm(dirn))

    def mindist2boundary(self, x):
        return np.amin(self.radius - np.linalg.norm(x - self.center, axis=-1))

    def boundary_normal(self, x):
        _n = x - self.center
        l = np.linalg.norm(_n, axis=-1, keepdims=True)
        _n = _n / l * np.isclose(l, self.radius)
        return _n

    def random_points(self, n, random="pseudo"):
        """https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
        """
        if random == "pseudo":
            U = np.random.rand(n, 1)
            X = np.random.normal(size=(n, self.dim))
        else:
            rng = sample(n, self.dim + 1, random)
            U, X = rng[:, 0:1], rng[:, 1:]
            X = stats.norm.ppf(X)
        X = preprocessing.normalize(X)
        X = U ** (1 / self.dim) * X
        return self.radius * X + self.center

    def random_boundary_points(self, n, random="pseudo"):
        """http://mathworld.wolfram.com/HyperspherePointPicking.html
        """
        if random == "pseudo":
            X = np.random.normal(size=(n, self.dim))
        else:
            U = sample(n, self.dim, random)
            X = stats.norm.ppf(U)
        X = preprocessing.normalize(X)
        return self.radius * X + self.center

    def background_points(self, x, dirn, dist2npt, shift):
        dirn = dirn / np.linalg.norm(dirn)
        dx = self.distance2boundary_unitdirn(x, -dirn)
        n = max(dist2npt(dx), 1)
        h = dx / n
        pts = x - np.arange(-shift, n - shift + 1)[:, None] * h * dirn
        return pts


class HyperEllipticalToroid(Geometry):
    """
        Class for parametric PINNs for toroidal shapes depending on three
        shape parameters and one parameter that controls the pressure profile,
        so number of inputs is 2 (X, Y) + 3 (eps, kappa, delta) + 1 (A) = 6.
    """
    def __init__(
        self,
        eps_range=(0.1, 0.3),
        kappa_range=(0.1, 0.3),
        delta_range=(0.1, 0.3),
        x_ellipse=[],
        Amax=0.1
    ):
        self.N = 100
        self.num_param = 2
        self.center = np.array(
            [[0.0, 0.0, 0.0,
              eps_range[1] - eps_range[0],
              kappa_range[1] - kappa_range[0],
              delta_range[1] - delta_range[0]
              ]]
        )
        # self.center, self.eps, self.kappa, self.delta = np.array(
        #     [[0.0, 0.0, 0.0]]), eps, kappa, delta
        self.tau = np.linspace(0, 2 * np.pi, self.N)
        Arange = np.linspace(-Amax, Amax, self.num_param)
        self.eps = np.linspace(eps_range[0], eps_range[1], self.num_param)
        self.kappa = np.linspace(kappa_range[0], kappa_range[1], self.num_param)
        self.delta = np.linspace(delta_range[0], delta_range[1], self.num_param)

        R_ellipse = np.zeros((self.N, self.num_param, self.num_param, self.num_param, self.num_param))
        Z_ellipse = np.zeros((self.N, self.num_param, self.num_param, self.num_param, self.num_param))
        A_ellipse = np.zeros((self.N, self.num_param, self.num_param, self.num_param, self.num_param))
        eps_ellipse = np.zeros((self.N, self.num_param, self.num_param, self.num_param, self.num_param))
        kappa_ellipse = np.zeros((self.N, self.num_param, self.num_param, self.num_param, self.num_param))
        delta_ellipse = np.zeros((self.N, self.num_param, self.num_param, self.num_param, self.num_param))
        for i in range(self.num_param):
            for j in range(self.num_param):
                for k in range(self.num_param):
                    for kk in range(self.num_param):
                        R_ellipse[:, i, j, k, kk] = 1 + self.eps[j] * np.cos(self.tau + np.arcsin(self.delta[kk]) * np.sin(self.tau))
                        Z_ellipse[:, i, j, k, kk] = self.eps[j] * self.kappa[k] * np.sin(self.tau)
                        A_ellipse[:, i, j, k, kk] = Arange[i]
                        eps_ellipse[:, i, j, k, kk] = self.eps[j]
                        kappa_ellipse[:, i, j, k, kk] = self.kappa[k]
                        delta_ellipse[:, i, j, k, kk] = self.delta[kk]

        # Define boundary of hyper-elliptical disk
        self.x_ellipse = np.transpose(
            np.asarray(
                [R_ellipse, Z_ellipse, A_ellipse,
                 eps_ellipse, kappa_ellipse, delta_ellipse]),
            [1, 2, 3, 4, 5, 0]
        )
        self.x_ellipse = self.x_ellipse.reshape(self.N * self.num_param ** 4, 6)

        # setting xmin and xmax for bbox
        xmin = np.array([1 - np.max(self.eps), -np.max(self.kappa * self.eps), -Amax, eps_range[0], kappa_range[0], delta_range[0]])
        xmax = np.array([1 + np.max(self.eps), np.max(self.kappa * self.eps), Amax, eps_range[-1], kappa_range[-1], delta_range[-1]])
        self.Amax = Amax

        super(HyperEllipticalToroid, self).__init__(6, (xmin,xmax), 1)

    def inside(self, x):
        return is_point_in_path(x[:, 0:1], x[:, 1:2], self.x_ellipse)

    def on_boundary(self, x):
        # This is not finding the distance of 2d points. Only for 1d does this work.
        return np.array([self.point_on_boundary(x[i]) for i in range(len(x))])

    def point_on_boundary(self, x):
        # Input
        #   x: A point i.e. array([1.0, 0.3])
        # Output
        #   True/False
        tol = np.max(np.linalg.norm(self.x_ellipse[:-1, 0:2] - self.x_ellipse[1:, 0:2], axis=-1))
        abs_diff = np.abs(x[:, 0:2] - self.x_ellipse[:, 0:2])
        return np.any(np.sqrt(abs_diff[:, 0:1]**2 + abs_diff[:, 1:2]**2) <= tol)
        # or np.allclose(abs(abs_diff[:, 2:3]), self.Amax)

    def random_points(self, n, random="pseudo"):
        x = []
        vbbox = self.bbox[1] - self.bbox[0]
        while len(x) < n:
            x_new = np.random.rand(1, 6) * vbbox + self.bbox[0]
            if self.inside(x_new):
                x.append(x_new)
        return np.vstack(x)

    def uniform_boundary_points(self, n):
        Arange = np.linspace(-self.Amax, self.Amax, self.num_param)
        tau = np.linspace(0, 2 * np.pi, n)
        R_ellipse = np.zeros((self.N, self.num_param, self.num_param, self.num_param, self.num_param))
        Z_ellipse = np.zeros((self.N, self.num_param, self.num_param, self.num_param, self.num_param))
        A_ellipse = np.zeros((self.N, self.num_param, self.num_param, self.num_param, self.num_param))
        eps_ellipse = np.zeros((self.N, self.num_param, self.num_param, self.num_param, self.num_param))
        kappa_ellipse = np.zeros((self.N, self.num_param, self.num_param, self.num_param, self.num_param))
        delta_ellipse = np.zeros((self.N, self.num_param, self.num_param, self.num_param, self.num_param))
        for i in range(self.num_param):
            for j in range(self.num_param):
                for k in range(self.num_param):
                    for kk in range(self.num_param):
                        R_ellipse[:, i, j, k, kk] = 1 + self.eps[j] * np.cos(tau + np.arcsin(self.delta[kk]) * np.sin(tau))
                        Z_ellipse[:, i, j, k, kk] = self.eps[j] * self.kappa[k] * np.sin(tau)
                        A_ellipse[:, i, j, k, kk] = Arange[i]
                        eps_ellipse[:, i, j, k, kk] = eps[j]
                        kappa_ellipse[:, i, j, k, kk] = kappa[k]
                        delta_ellipse[:, i, j, k, kk] = delta[kk]
        X = np.transpose(
            np.asarray(
                [R_ellipse, Z_ellipse, A_ellipse,
                 eps_ellipse, kappa_ellipse, delta_ellipse]),
            [1, 2, 3, 4, 5, 0]).reshape(n * self.num_param ** 4, 6)
        return X

    def random_boundary_points(self, n, random="pseudo"):
        u = sample(n, 1, random)
        tau = 2 * np.pi * u
        Arange = (sample(self.num_param, 1, random) - 0.5) * 2 * self.Amax
        eps = (sample(self.num_param, 1, random) + self.eps[-1] - self.eps[0]) / self.eps[-1]
        kappa = (sample(self.num_param, 1, random) + self.kappa[-1] - self.kappa[0]) / self.kappa[-1]
        delta = (sample(self.num_param, 1, random) + self.delta[-1] - self.delta[0]) / self.delta[-1]
        R_ellipse = np.zeros((self.N, self.num_param, self.num_param, self.num_param, self.num_param))
        Z_ellipse = np.zeros((self.N, self.num_param, self.num_param, self.num_param, self.num_param))
        A_ellipse = np.zeros((self.N, self.num_param, self.num_param, self.num_param, self.num_param))
        eps_ellipse = np.zeros((self.N, self.num_param, self.num_param, self.num_param, self.num_param))
        kappa_ellipse = np.zeros((self.N, self.num_param, self.num_param, self.num_param, self.num_param))
        delta_ellipse = np.zeros((self.N, self.num_param, self.num_param, self.num_param, self.num_param))
        for i in range(self.num_param):
            for j in range(self.num_param):
                for k in range(self.num_param):
                    for kk in range(self.num_param):
                        R_ellipse[:, i, j, k, kk] = 1 + eps[j] * np.cos(tau + np.arcsin(delta[kk]) * np.sin(tau))
                        Z_ellipse[:, i, j, k, kk] = eps[j] * kappa[k] * np.sin(tau)
                        A_ellipse[:, i, j, k, kk] = Arange[i]
                        eps_ellipse[:, i, j, k, kk] = eps[j]
                        kappa_ellipse[:, i, j, k, kk] = kappa[k]
                        delta_ellipse[:, i, j, k, kk] = delta[kk]
        X = np.transpose(
            np.asarray(
                [R_ellipse, Z_ellipse, A_ellipse,
                 eps_ellipse, kappa_ellipse, delta_ellipse]),
            [1, 2, 3, 4, 5, 0]).reshape(n * self.num_param ** 4, 6)
        return X


def is_point_in_path(x, y, poly) -> bool:
    num = len(poly)
    j = num - 1
    c = False
    for i in range(num):
        # if abs(A) == Amax:
        #     return True  # on the [-Amax, Amax] boundary surface
        if (x == poly[i][0]) and (y == poly[i][1]):  # and (A == poly[i][2]):
            # point is a corner
            return True
        if ((poly[i][1] > y) != (poly[j][1] > y)):
            slope = (x-poly[i][0])*(poly[j][1]-poly[i][1])-(poly[j][0]-poly[i][0])*(y-poly[i][1])
            if slope == 0:
                # point is on boundary
                return True
            if (slope < 0) != (poly[j][1] < poly[i][1]):
                c = not c
        j = i
    return c
