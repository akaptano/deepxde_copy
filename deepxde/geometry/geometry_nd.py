import itertools

import numpy as np
from scipy import stats
from sklearn import preprocessing

from .geometry import Geometry
from .sampler import sample
from .. import config


class Hypercube(Geometry):
    def __init__(self, xmin, xmax):
        if len(xmin) != len(xmax):
            raise ValueError("Dimensions of xmin and xmax do not match.")

        self.xmin = np.array(xmin, dtype=config.real(np))
        self.xmax = np.array(xmax, dtype=config.real(np))
        if np.any(self.xmin >= self.xmax):
            raise ValueError("xmin >= xmax")

        self.side_length = self.xmax - self.xmin
        super().__init__(
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
        _n = -np.isclose(x, self.xmin).astype(config.real(np)) + np.isclose(
            x, self.xmax
        )
        # For vertices, the normal is averaged for all directions
        idx = np.count_nonzero(_n, axis=-1) > 1
        if np.any(idx):
            print(
                f"Warning: {self.__class__.__name__} boundary_normal called on vertices. "
                "You may use PDE(..., exclusions=...) to exclude the vertices."
            )
            l = np.linalg.norm(_n[idx], axis=-1, keepdims=True)
            _n[idx] /= l
        return _n

    def uniform_points(self, n, boundary=True):
        dx = (self.volume / n) ** (1 / self.dim)
        xi = []
        for i in range(self.dim):
            ni = int(np.ceil(self.side_length[i] / dx))
            if boundary:
                xi.append(
                    np.linspace(
                        self.xmin[i], self.xmax[i], num=ni, dtype=config.real(np)
                    )
                )
            else:
                xi.append(
                    np.linspace(
                        self.xmin[i],
                        self.xmax[i],
                        num=ni + 1,
                        endpoint=False,
                        dtype=config.real(np),
                    )[1:]
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

    def random_boundary_points(self, n, random="pseudo"):
        x = sample(n, self.dim, random)
        # Randomly pick a dimension
        rand_dim = np.random.randint(self.dim, size=n)
        # Replace value of the randomly picked dimension with the nearest boundary value (0 or 1)
        x[np.arange(n), rand_dim] = np.round(x[np.arange(n), rand_dim])
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
        self.center = np.array(center, dtype=config.real(np))
        self.radius = radius
        super().__init__(
            len(center), (self.center - radius, self.center + radius), 2 * radius
        )

        self._r2 = radius ** 2

    def inside(self, x):
        return np.linalg.norm(x - self.center, axis=-1) <= self.radius

    def on_boundary(self, x):
        return np.isclose(np.linalg.norm(x - self.center, axis=-1), self.radius)

    def distance2boundary_unitdirn(self, x, dirn):
        # https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
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
        # https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
        if random == "pseudo":
            U = np.random.rand(n, 1)
            X = np.random.normal(size=(n, self.dim))
        else:
            rng = sample(n, self.dim + 1, random)
            U, X = rng[:, 0:1], rng[:, 1:]  # Error if X = [0, 0, ...]
            X = stats.norm.ppf(X)
        X = preprocessing.normalize(X)
        X = U ** (1 / self.dim) * X
        return self.radius * X + self.center

    def random_boundary_points(self, n, random="pseudo"):
        # http://mathworld.wolfram.com/HyperspherePointPicking.html
        if random == "pseudo":
            X = np.random.normal(size=(n, self.dim)).astype(config.real(np))
        else:
            U = sample(n, self.dim, random)  # Error for [0, 0, ...] or [0.5, 0.5, ...]
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
        Amax=0.1,
        num_param=2,
    ):
        self.N = 100
        self.num_param = num_param
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


class HyperFourierEllipse(Geometry):
    """
        Class for parametric PINNs for toroidal shapes depending on mpol
        shape parameters and one parameter that controls the pressure profile,
        so number of inputs is
        2 (X, Y) + 1 (A) + 2 * mpol (R(0, 1), ..., R(0, mpol), Z(0, 1), ..., Z(0, mpol))
    """
    def __init__(
        self,
        mpol=1,
        x_ellipse=[],
        Amax=0.1,
        num_param=2,
        RZm_max=0.1,
        minor_radius=0.8
    ):
        if minor_radius >= 1.0:
            raise ValueError(
                'Minor radius must be less than 1, since the major radius'
                ' is hard-coded to this value'
            )
        self.N = 100
        self.num_param = num_param
        self.num_dims = 3 + 2 * mpol
        self.tau = np.linspace(0, 2 * np.pi, self.N)
        self.minor_radius = minor_radius

        # begin with perfectly circular cross-section
        self.center = np.zeros((3 + 2 * mpol))

        Arange = np.linspace(-Amax, Amax, self.num_param)
        Rm = np.linspace(-RZm_max, RZm_max, self.num_param)
        Zm = np.linspace(-RZm_max, RZm_max, self.num_param)

        Rm_grid = Rm
        Zm_grid = Zm
        for i in range(mpol + 1):
            Rm_grid = np.outer(Rm_grid, Rm).reshape(2 * mpol * np.ones(i + 2, dtype=int))
            Zm_grid = np.outer(Zm_grid, Zm).reshape(2 * mpol * np.ones(i + 2, dtype=int))
        # Rm_grid = [Rm] * mpol
        print(Rm_grid.shape)

        # assume that minor radius R(0, 0) = 1
        R_ellipse = np.ones((self.N, self.num_param, *Rm_grid.shape))
        R0 = np.outer(
            self.minor_radius * np.cos(self.tau),
            np.ravel(np.ones((self.num_param, *Rm_grid.shape)))
            ).reshape(R_ellipse.shape)
        R_ellipse = (1 + R0) * R_ellipse

        # Z(0, 0) = sin(tau)
        Z_ellipse = np.ones((self.N, self.num_param, *Rm_grid.shape))
        Z0 = np.outer(
            self.minor_radius * np.sin(self.tau),
            np.ravel(np.ones((self.num_param, *Rm_grid.shape)))
            ).reshape(Z_ellipse.shape)
        Z_ellipse = Z0 * Z_ellipse

        A_ellipse = np.zeros((self.N, self.num_param, *Rm_grid.shape))

        for i in range(self.num_param):
            for m in range(1, mpol + 1):  # sum over m
                R_ellipse[:, i, ...] += np.outer(
                    np.cos(m * self.tau),
                    np.ravel(Rm_grid)
                    ).reshape(self.N, *Rm_grid.shape)
                Z_ellipse[:, i, ...] += np.outer(
                    np.sin(m * self.tau),
                    np.ravel(Zm_grid)
                    ).reshape(self.N, *Rm_grid.shape)
            A_ellipse[:, i, ...] = Arange[i]

        self.R_ellipse = R_ellipse
        self.Z_ellipse = Z_ellipse

        Rm_grid = np.ones((self.N, self.num_param, *Rm_grid.shape)) * Rm_grid
        Zm_grid = np.ones((self.N, self.num_param, *Zm_grid.shape)) * Zm_grid

        # Define boundary of hyper-cross-section
        inds = np.roll(np.arange(0, len(Rm_grid.shape) + 1), -1)
        x_ellipse = np.transpose(
            np.array([R_ellipse, Z_ellipse, A_ellipse]),
            inds
        )
        start = 0
        end = self.num_param
        for i in range(mpol):
            Rm_grid_partial = np.ones(Rm_grid.shape)
            Zm_grid_partial = np.ones(Zm_grid.shape)
            slc = [slice(None)] * len(Rm_grid.shape)
            slc[2 + i] = slice(start, end)
            Rm_grid_partial[slc] = Rm
            Zm_grid_partial[slc] = Zm
            Rm_grid_partial = Rm_grid_partial.reshape(*Rm_grid_partial.shape, 1)
            Zm_grid_partial = Zm_grid_partial.reshape(*Zm_grid_partial.shape, 1)
            x_ellipse = np.concatenate([x_ellipse, Rm_grid_partial], axis=-1)
            x_ellipse = np.concatenate([x_ellipse, Zm_grid_partial], axis=-1)

        print(x_ellipse.shape, Rm_grid.shape)
        self.x_ellipse = x_ellipse.reshape(
            self.N * self.num_param ** (1 + 2 * mpol),
            3 + 2 * mpol
        )

        # setting xmin and xmax for bbox
        xmin = np.array([
            1 - self.minor_radius - RZm_max,
            -1 - RZm_max - self.minor_radius,
            -Amax])
        xmax = np.array([
            1 + self.minor_radius + RZm_max,
            1 + RZm_max + self.minor_radius,
            Amax])

        for m in range(2 * mpol):
            xmin = np.concatenate((xmin, [-RZm_max]))
            xmax = np.concatenate((xmax, [RZm_max]))

        self.Amax = Amax
        self.RZm_max = RZm_max

        super(HyperFourierEllipse, self).__init__(3 + 2 * mpol, (xmin, xmax), 1)

    def inside(self, x):
        return (np.sqrt((x[:, 0:1] - 1.0) ** 2 + x[:, 1:2] ** 2) < self.minor_radius)
        # return is_point_in_path(x[:, 0:1], x[:, 1:2], self.x_ellipse)

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
            x_new = np.random.rand(1, self.num_dims) * vbbox + self.bbox[0]
            if self.inside(x_new):
                x.append(x_new)
        return np.vstack(x)

    def uniform_boundary_points(self, n):
        RZm_max = self.RZm_max
        tau = np.linspace(0, 2 * np.pi, self.N)
        Arange = np.linspace(-Amax, Amax, self.num_param)
        Rm = np.linspace(-RZm_max, RZm_max, self.num_param)
        Zm = np.linspace(-RZm_max, RZm_max, self.num_param)

        Rm_grid = Rm
        Zm_grid = Zm
        for i in range(mpol + 1):
            Rm_grid = np.outer(Rm_grid, Rm).reshape(2 * mpol * np.ones(i + 2, dtype=int))
            Zm_grid = np.outer(Zm_grid, Zm).reshape(2 * mpol * np.ones(i + 2, dtype=int))

        # assume that minor radius R(0, 0) = 1
        R_ellipse = np.ones((n, self.num_param, *Rm_grid.shape))
        R0 = np.outer(
            self.minor_radius * np.cos(self.tau),
            np.ravel(np.ones((self.num_param, *Rm_grid.shape)))
            ).reshape(R_ellipse.shape)
        R_ellipse = (1 + R0) * R_ellipse

        # Z(0, 0) = 0
        Z_ellipse = np.ones((self.N, self.num_param, *Rm_grid.shape))
        Z0 = np.outer(
            self.minor_radius * np.sin(self.tau),
            np.ravel(np.ones((self.num_param, *Rm_grid.shape)))
            ).reshape(Z_ellipse.shape)
        Z_ellipse = Z0 * Z_ellipse

        A_ellipse = np.zeros((n, self.num_param, *Rm_grid.shape))

        for i in range(self.num_param):
            for m in range(1, mpol + 1):  # sum over m
                R_ellipse[:, i, ...] += np.outer(np.cos(m * self.tau), Rm_grid).reshape(n, *Rm_grid.shape)
                Z_ellipse[:, i, ...] += np.outer(np.sin(m * self.tau), Zm_grid).reshape(n, *Rm_grid.shape)
            A_ellipse[:, i, ...] = Arange[i]

        Rm_grid = np.ones((n, self.num_param, *Rm_grid.shape)) * Rm_grid
        Zm_grid = np.ones((n, self.num_param, *Zm_grid.shape)) * Zm_grid

        # Define boundary of hyper-cross-section
        inds = np.roll(np.arange(0, len(Rm_grid.shape) + 1), -1)
        X = np.transpose(
            np.array([R_ellipse, Z_ellipse, A_ellipse]),
            inds
        )
        start = 0
        end = self.num_param
        for i in range(mpol):
            Rm_grid_partial = np.ones(Rm_grid.shape)
            Zm_grid_partial = np.ones(Zm_grid.shape)
            slc = [slice(None)] * len(Rm_grid.shape)
            slc[2 + i] = slice(start, end)
            Rm_grid_partial[slc] = Rm
            Zm_grid_partial[slc] = Zm
            Rm_grid_partial = Rm_grid_partial.reshape(*Rm_grid_partial.shape, 1)
            Zm_grid_partial = Zm_grid_partial.reshape(*Zm_grid_partial.shape, 1)
            X = np.concatenate([X, Rm_grid_partial], axis=-1)
            X = np.concatenate([X, Zm_grid_partial], axis=-1)

        inds = np.roll(np.arange(0, len(x_ellipse.shape)), -1)
        X = np.transpose(
            X,
            inds
        )
        X = X.reshape(
            n * self.num_param ** (1 + 2 * mpol),
            3 + 2 * mpol
        )
        return X

    def random_boundary_points(self, n, random="pseudo"):
        u = sample(n, 1, random)
        tau = 2 * np.pi * u
        RZm_max = self.RZm_max
        Arange = (sample(self.num_param, 1, random) - 0.5) * 2 * self.Amax
        Rm = (sample(self.num_param, 1, random) - 0.5) * 2 * RZm_max
        Zm = (sample(self.num_param, 1, random) - 0.5) * 2 * RZm_max

        Rm_grid = Rm
        Zm_grid = Zm
        for i in range(mpol + 1):
            Rm_grid = np.outer(Rm_grid, Rm).reshape(2 * mpol * np.ones(i + 2, dtype=int))
            Zm_grid = np.outer(Zm_grid, Zm).reshape(2 * mpol * np.ones(i + 2, dtype=int))

        # assume that major radius R(0, 0) = 1
        R_ellipse = np.ones((n, self.num_param, *Rm_grid.shape))
        R0 = np.outer(
            self.minor_radius * np.cos(self.tau),
            np.ravel(np.ones((self.num_param, *Rm_grid.shape)))
            ).reshape(R_ellipse.shape)
        R_ellipse = (1 + R0) * R_ellipse

        # Z(0, 0) = 0
        Z_ellipse = np.ones((self.N, self.num_param, *Rm_grid.shape))
        Z0 = np.outer(
            self.minor_radius * np.sin(self.tau),
            np.ravel(np.ones((self.num_param, *Rm_grid.shape)))
            ).reshape(Z_ellipse.shape)
        Z_ellipse = Z0 * Z_ellipse

        A_ellipse = np.zeros((n, self.num_param, *Rm_grid.shape))

        for i in range(self.num_param):
            for m in range(1, mpol + 1):  # sum over m
                R_ellipse[:, i, ...] += np.outer(np.cos(m * self.tau), Rm_grid).reshape(n, *Rm_grid.shape)
                Z_ellipse[:, i, ...] += np.outer(np.sin(m * self.tau), Zm_grid).reshape(n, *Rm_grid.shape)
            A_ellipse[:, i, ...] = Arange[i]

        Rm_grid = np.ones((n, self.num_param, *Rm_grid.shape)) * Rm_grid
        Zm_grid = np.ones((n, self.num_param, *Zm_grid.shape)) * Zm_grid

        # Define boundary of hyper-cross-section
        inds = np.roll(np.arange(0, len(Rm_grid.shape) + 1), -1)
        X = np.transpose(
            np.array([R_ellipse, Z_ellipse, A_ellipse]),
            inds
        )
        start = 0
        end = self.num_param
        for i in range(mpol):
            Rm_grid_partial = np.ones(Rm_grid.shape)
            Zm_grid_partial = np.ones(Zm_grid.shape)
            slc = [slice(None)] * len(Rm_grid.shape)
            slc[2 + i] = slice(start, end)
            Rm_grid_partial[slc] = Rm
            Zm_grid_partial[slc] = Zm
            Rm_grid_partial = Rm_grid_partial.reshape(*Rm_grid_partial.shape, 1)
            Zm_grid_partial = Zm_grid_partial.reshape(*Zm_grid_partial.shape, 1)
            X = np.concatenate([X, Rm_grid_partial], axis=-1)
            X = np.concatenate([X, Zm_grid_partial], axis=-1)

        inds = np.roll(np.arange(0, len(x_ellipse.shape)), -1)
        X = np.transpose(
            X,
            inds
        )
        X = X.reshape(
            n * self.num_param ** (1 + 2 * mpol),
            3 + 2 * mpol
        )
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
