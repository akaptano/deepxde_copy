import itertools

import numpy as np
from scipy import stats
from sklearn import preprocessing
from .geometry import Geometry
from .sampler import sample
from .. import config
import copy

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

        # a uniform grid of points for each parameter in each dimension (or, say, in a one dimensional setting) on the boundary
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
        # Reshape the array by transposing it and then reshaping it
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
        2 (X, Y) + 1 (A) + 2 * mpol (R_1, ..., R_mpol, Z_1, ..., Z_mpol) + 1 (R_0) = 4 + 2 * mpol
    """
    def __init__(
        self,
        mpol=1,     # number of Fourier modes
        x_ellipse=[],
        Amax=0.1,    # maximum value of A
        num_param=2, # number of parameters per dimension
        RZm_max=0.1, # maximum absolute value of R and Z
        minor_radius=0.8 # minor radius
    ):
        if minor_radius >= 1.0:
            raise ValueError(
                'Minor radius must be less than 1, since the major radius'
                ' is hard-coded to this value'
            )
        self.N = 100  # number of collocation points on the boundary?????
        self.num_param = num_param  # number of parameters per dimension
        self.num_dims = 3 + 2 * mpol  # number of dimensions, in this case R, Z, A, and Fourier modes
        self.tau = np.linspace(0, 2 * np.pi, self.N)  # values for theta
        self.minor_radius = minor_radius  
        self.mpol = mpol  # number of Fourier modes
        self.Amax = Amax
        self.RZm_max = RZm_max


        # begin with perfectly circular cross-section
        self.center = np.zeros((self.num_dims))     # center of the ellipse, should have the same number of dimensions as self.num_dims


        # This is a uniform initialization of R, Z, A, and the Fourier coefficients Rm, Zm
        # Define x_ellipse as a holder for [R, Z, A, Rm, Zm], 
        # where dimensions of Rm and Zm depends on mpol
        # This is what we want to plug into x_ellipse
        x_ellipse = []

        Arange = np.linspace(-self.Amax, self.Amax, self.num_param)  # range of A 
        Rm = np.linspace(-self.RZm_max, self.RZm_max, self.num_param)  # range of Rm
        Zm = np.linspace(-self.RZm_max, self.RZm_max, self.num_param)  # range of Zm

        Rm_grid = Rm
        Zm_grid = Zm

        shape = (self.N,) + (self.num_param,) * (mpol*2+1)  # (100, 4, 4, 4, 4, 4)
        R_ellipse = np.ones(shape)
        Z_ellipse = np.ones(shape)
        A_ellipse = np.ones(shape)
        # R_ellipse = np.ones(shape) * self.minor_radius
        # Z_ellipse = np.ones(shape) * self.minor_radius
        # A_ellipse = np.ones(shape) * self.minor_radius

        # We have to handle Rm and Zm first, so that we can calculate R_ellipse and Z_ellipse using them

        # Initialize base arrays for Fourier coefficients
        # R0 is 1 (nonzero) and Z0 is 0 for the base circular shape
        R0 = np.ones(shape) * self.minor_radius  # R0 coefficient is 1 for circular base shape
        Z0 = np.zeros(shape) * self.minor_radius # Z0 coefficient is 0 for circular base shape

        # Initialize arrays for higher order Fourier coefficients
        Rm_coeffs = []
        Zm_coeffs = []
        
        # For each Fourier mode m, create coefficient arrays
        for m in range(1, mpol + 1):
            # Create coefficient arrays with proper shape
            Rm_m = np.ones(shape) * Rm_grid
            Zm_m = np.ones(shape) * Zm_grid
            # Rm_m = np.ones(shape) * self.minor_radius * Rm_grid
            # Zm_m = np.ones(shape) * self.minor_radius * Zm_grid
            Rm_coeffs.append(Rm_m)
            Zm_coeffs.append(Zm_m)


        # Stack all coefficients together
        # shape for Rm_grid: (100, 4, 4, 4, 4, 4, 2)
        # 100: number of points in tau
        # 4, 4, 4, 4, 4: parameter dimensions (5 trainable variables, with 4 instances for each)
        # 2: Fourier modes (m=1,2)
        # We want to fix the first element R0 as SELF.minor_radius and Z0 as zeros
        # Rm_grid = np.stack([R0] + Rm_coeffs, axis=-1)  # delete the first dimension
        # Zm_grid = np.stack([Z0] + Zm_coeffs, axis=-1)  # delete the first dimension
        Rm_grid = np.stack(Rm_coeffs, axis=-1)
        Zm_grid = np.stack(Zm_coeffs, axis=-1)


        # Create indices for all combinations instead of using nested loops
        # mpol*2+1 is the number for parameters A, Rm, Zm
        indices = np.indices((self.num_param,) * (mpol*2+1)).reshape(mpol*2+1, -1).T
        # print("indices.shape", indices.shape)  # (1024, 5) = (4^5, 5)

        # Now we can calculate R_ellipse and Z_ellipse using the indices

        for idx in indices:
            slc = (slice(None),) + tuple(idx)   # (slice(None, None, None), 0, 0, 0, 0, 0)
            # print("np.cos(m * self.tau).shape", np.cos(m * self.tau).shape)  # (100,)
            # print("Rm_grid.shape", Rm_grid.shape)  # (100, 4, 4, 4, 4, 4, 2)
            # print("Rm_grid[slc].shape", Rm_grid[slc].shape)   # (100, 2)
            # print("Rm_grid[slc][:, m].shape", Rm_grid[slc][:, 1].shape)  # (100,)
            # print(np.array([np.multiply(Rm_grid[slc][:, m], np.cos(m * self.tau)) for m in range(1, Rm_grid.shape[-1])]).shape)  # (1, 100)
            R_ellipse[slc] = np.sum([np.multiply(Rm_grid[slc][:, m], np.cos(m * self.tau)) for m in range(1, Rm_grid.shape[-1])], axis=0) 
            Z_ellipse[slc] = np.sum([np.multiply(Zm_grid[slc][:, m], np.sin(m * self.tau)) for m in range(1, Zm_grid.shape[-1])], axis=0) 
            A_ellipse[slc] = Arange[idx[0]]


        R_ellipse = R_ellipse + np.ones_like(R_ellipse) * self.minor_radius

        # Store the components
        self.R_ellipse = R_ellipse
        self.Z_ellipse = Z_ellipse
        self.A_ellipse = A_ellipse
        self.Rm_ellipse = Rm_grid
        self.Zm_ellipse = Zm_grid


        # Split Rm_grid and Zm_grid along last axis dynamically
        Rm_components = np.moveaxis(Rm_grid, -1, 0)  # Move last axis to first, shape: (mpol+1, 100, 4, 4, 4, 4, 4)
        Zm_components = np.moveaxis(Zm_grid, -1, 0)  # Move last axis to first, shape: (mpol+1, 100, 4, 4, 4, 4, 4)


        # Combine R_ellipse, Z_ellipse, A_ellipse, Rm_components and Zm_components 
        # x_ellipse has shape (100, 4, 4, 4, 4, 4, 7)
        x_ellipse = np.stack([self.R_ellipse, self.Z_ellipse, self.A_ellipse] + 
                         [comp for comp in Rm_components] + 
                         [comp for comp in Zm_components], axis=-1)

        x_ellipse = x_ellipse.reshape(self.N * self.num_param ** (1 + 2 * self.mpol), 3 + 2 * self.mpol)

        self.x_ellipse = x_ellipse


        # setting xmin and xmax for bounding box
        self.xmin = np.array([
            1 - self.minor_radius - self.RZm_max,
            -1 - self.RZm_max - self.minor_radius,
            -self.Amax])
        self.xmax = np.array([
            1 + self.minor_radius + self.RZm_max,
            1 + self.RZm_max + self.minor_radius,
            self.Amax])


        # Add bounds for Fourier coefficients
        for m in range(2 * mpol):
            self.xmin = np.concatenate((self.xmin, [-self.RZm_max]))
            self.xmax = np.concatenate((self.xmax, [self.RZm_max]))


        super(HyperFourierEllipse, self).__init__(4 + 2 * mpol, (self.xmin, self.xmax), 1)


    # change this!
    def inside(self, x):
        # print("x.shape", x.shape)   # x.shape (1, 7)
        # return (np.sqrt((x[:, 0:1] - 1.0) ** 2 + x[:, 1:2] ** 2) < self.minor_radius)
        return is_point_in_path_fourier(x[:, 0:1], x[:, 1:2], self.x_ellipse)

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
        print("RANDOM POINTS")
        x = []
        vbbox = self.xmax - self.xmin
        while len(x) < n:
            x_new = np.random.rand(1, self.num_dims) * vbbox + self.xmin
            if self.inside(x_new):
                x.append(x_new)
        return np.vstack(x)

    def uniform_boundary_points(self, n):   # the parameter n seems to be unused??
        # The init function alreay is a uniform initialization of R, Z, A, and the Fourier coefficients Rm, Zm
        print("UNIFORM BOUNDARY POINTS")

        self.N = n  # number of collocation points on the boundary?????

        # begin with perfectly circular cross-section
        self.center = np.zeros((self.num_dims))     # center of the ellipse, should have the same number of dimensions as self.num_dims


        # This is a uniform initialization of R, Z, A, and the Fourier coefficients Rm, Zm
        # Define x_ellipse as a holder for [R, Z, A, Rm, Zm], 
        # where dimensions of Rm and Zm depends on mpol
        # This is what we want to plug into x_ellipse
        x_ellipse = []

        Arange = np.linspace(-self.Amax, self.Amax, self.num_param)  # range of A 
        Rm = np.linspace(-self.RZm_max, self.RZm_max, self.num_param)  # range of Rm
        Zm = np.linspace(-self.RZm_max, self.RZm_max, self.num_param)  # range of Zm

        Rm_grid = Rm
        Zm_grid = Zm

        shape = (self.N,) + (self.num_param,) * (self.mpol*2+1)  # (100, 4, 4, 4, 4, 4)
        R_ellipse = np.ones(shape) * self.minor_radius
        Z_ellipse = np.ones(shape) * self.minor_radius
        A_ellipse = np.ones(shape) * self.minor_radius

        # We have to handle Rm and Zm first, so that we can calculate R_ellipse and Z_ellipse using them

        # Initialize base arrays for Fourier coefficients
        # R0 is 1 (nonzero) and Z0 is 0 for the base circular shape
        R0 = np.ones(shape) * self.minor_radius  # R0 coefficient is 1 for circular base shape
        Z0 = np.zeros(shape) * self.minor_radius # Z0 coefficient is 0 for circular base shape

        # Initialize arrays for higher order Fourier coefficients
        Rm_coeffs = []
        Zm_coeffs = []
        
        # For each Fourier mode m, create coefficient arrays
        for m in range(1, self.mpol + 1):
            # Create coefficient arrays with proper shape
            Rm_m = np.ones(shape) * self.minor_radius * Rm_grid
            Zm_m = np.ones(shape) * self.minor_radius * Zm_grid
            Rm_coeffs.append(Rm_m)
            Zm_coeffs.append(Zm_m)


        Rm_grid = np.stack(Rm_coeffs, axis=-1)
        Zm_grid = np.stack(Zm_coeffs, axis=-1)


        indices = np.indices((self.num_param,) * (self.mpol*2+1)).reshape(self.mpol*2+1, -1).T

        # Now we can calculate R_ellipse and Z_ellipse using the indices

        for idx in indices:
            slc = (slice(None),) + tuple(idx)
            R_ellipse[slc] = np.sum([np.multiply(Rm_grid[slc][:, m-1], np.cos(m * self.tau)) for m in range(1, Rm_grid.shape[-1])], axis=0) 
            Z_ellipse[slc] = np.sum([np.multiply(Zm_grid[slc][:, m-1], np.sin(m * self.tau)) for m in range(1, Zm_grid.shape[-1])], axis=0) 
            A_ellipse[slc] = Arange[idx[0]]

        R_ellipse = R_ellipse + np.ones_like(R_ellipse) * self.minor_radius

        # Store the components
        self.R_ellipse = R_ellipse
        self.Z_ellipse = Z_ellipse
        self.A_ellipse = A_ellipse
        self.Rm_ellipse = Rm_grid
        self.Zm_ellipse = Zm_grid

        Rm_components = np.moveaxis(Rm_grid, -1, 0)  # Move last axis to first, shape: (mpol+1, 100, 4, 4, 4, 4, 4)
        Zm_components = np.moveaxis(Zm_grid, -1, 0)  # Move last axis to first, shape: (mpol+1, 100, 4, 4, 4, 4, 4)


        x_ellipse = np.stack([R_ellipse, Z_ellipse, A_ellipse] + 
                         [comp for comp in Rm_components] + 
                         [comp for comp in Zm_components], axis=-1)

        x_ellipse = x_ellipse.reshape(self.N * self.num_param ** (1 + 2 * self.mpol), 3 + 2 * self.mpol)
        

        return x_ellipse

    def random_boundary_points(self, n, random="pseudo"):
        # sample(n_samples, dimension, "something") returns an array of shape (n_samples, dimension)

        tau = 2 * np.pi * sample(n, 1, random)
        Arange = (sample(self.num_param, 1, random) - 0.5) * 2 * self.Amax
        Rm = (sample(self.num_param, 1, random) - 0.5) * 2 * self.RZm_max
        Zm = (sample(self.num_param, 1, random) - 0.5) * 2 * self.RZm_max

        Rm_grid = Rm
        Zm_grid = Zm

        shape = (self.N,) + (self.num_param,) * (self.mpol*2+1)  # (100, 4, 4, 4, 4, 4)
        R_ellipse = np.ones(shape) * self.minor_radius
        Z_ellipse = np.ones(shape) * self.minor_radius
        A_ellipse = np.ones(shape) * self.minor_radius


        # We have to handle Rm and Zm first, so that we can calculate R_ellipse and Z_ellipse using them

        # Initialize base arrays for Fourier coefficients
        # R0 is 1 (nonzero) and Z0 is 0 for the base circular shape
        R0 = np.ones(shape) * self.minor_radius  # R0 coefficient is 1 for circular base shape
        Z0 = np.zeros(shape) * self.minor_radius # Z0 coefficient is 0 for circular base shape

        # Initialize arrays for higher order Fourier coefficients
        Rm_coeffs = []
        Zm_coeffs = []
        
        # For each Fourier mode m, create coefficient arrays
        for m in range(1, self.mpol + 1):
            # Create coefficient arrays with proper shape
            Rm_m = np.ones(shape) * self.minor_radius * Rm_grid
            Zm_m = np.ones(shape) * self.minor_radius * Zm_grid
            Rm_coeffs.append(Rm_m)
            Zm_coeffs.append(Zm_m)


        Rm_grid = np.stack(Rm_coeffs, axis=-1)
        Zm_grid = np.stack(Zm_coeffs, axis=-1)


        # Create indices for all combinations instead of using nested loops
        # mpol*2+1 is the number for parameters A, Rm, Zm
        indices = np.indices((self.num_param,) * (self.mpol*2+1)).reshape(self.mpol*2+1, -1).T
        # print("indices.shape", indices.shape)  # (1024, 5) = (4^5, 5)

        # Now we can calculate R_ellipse and Z_ellipse using the indices

        for idx in indices:
            slc = (slice(None),) + tuple(idx)   # (slice(None, None, None), 0, 0, 0, 0, 0)
            R_ellipse[slc] = np.sum([np.multiply(Rm_grid[slc][:, m], np.cos(m * self.tau)) for m in range(1, Rm_grid.shape[-1])], axis=0) 
            Z_ellipse[slc] = np.sum([np.multiply(Zm_grid[slc][:, m], np.sin(m * self.tau)) for m in range(1, Zm_grid.shape[-1])], axis=0) 
            A_ellipse[slc] = Arange[idx[0]]

        R_ellipse = R_ellipse + np.ones_like(R_ellipse) * self.minor_radius

        # Store the components
        self.R_ellipse = R_ellipse
        self.Z_ellipse = Z_ellipse
        self.A_ellipse = A_ellipse
        self.Rm_ellipse = Rm_grid
        self.Zm_ellipse = Zm_grid

        x_ellipse = np.concatenate([R_ellipse, Z_ellipse, A_ellipse], axis=-1)
        # Split Rm_grid and Zm_grid along last axis dynamically
        Rm_components = np.moveaxis(Rm_grid, -1, 0)  # Move last axis to first, shape: (mpol+1, 100, 4, 4, 4, 4, 4)
        Zm_components = np.moveaxis(Zm_grid, -1, 0)  # Move last axis to first, shape: (mpol+1, 100, 4, 4, 4, 4, 4)


        x_ellipse = np.stack([R_ellipse, Z_ellipse, A_ellipse] + 
                         [comp for comp in Rm_components] + 
                         [comp for comp in Zm_components], axis=-1)

        x_ellipse = x_ellipse.reshape(self.N * self.num_param ** (1 + 2 * self.mpol), 3 + 2 * self.mpol)

        return x_ellipse


# Test if this works

# This is the Ray casting algorithm
# https://en.wikipedia.org/wiki/Point_in_polygon#:~:text=Ray%20casting%20algorithm,-See%20also%3A%20Jordan&text=The%20number%20of%20intersections%20for,also%20works%20in%20three%20dimensions.

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



def is_point_in_path_fourier(x, y, boundary_points) -> bool:
    """Determines if a point (x,y) is inside a boundary defined by Fourier series points.
    Uses a vectorized ray casting algorithm for better efficiency.
    
    Args:
        x: x-coordinate of the point to test
        y: y-coordinate of the point to test 
        boundary_points: Array of boundary points generated by Fourier series
        
    Returns:
        bool: True if point is inside or on the boundary, False otherwise
    """
    # Convert inputs to numpy arrays if not already
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    
    # Get next point indices (wrapping around to start)
    next_idx = np.roll(np.arange(len(boundary_points)), -1)
    
    # Extract current and next point coordinates
    y1 = boundary_points[:, 1]
    y2 = boundary_points[next_idx, 1]
    x1 = boundary_points[:, 0]
    x2 = boundary_points[next_idx, 0]
    
    # Vectorized check for y-range
    mask = ((y1 <= y) & (y2 > y)) | ((y2 <= y) & (y1 > y))
    
    # Calculate x-intersections where mask is True
    x_ints = x1[mask] + (y - y1[mask]) * (x2[mask] - x1[mask]) / (y2[mask] - y1[mask])
    
    # Count intersections to the right of point x
    num_intersections = np.sum(x_ints > x)

    # print(f"Point ({x}, {y}): intersections = {num_intersections}")
    
    return (num_intersections % 2) == 1



# # Test with a simple circle
# theta = np.linspace(0, 2*np.pi, 100)
# r = 1.0
# boundary_points = np.column_stack((r*np.cos(theta), r*np.sin(theta)))

# # Test points
# test_points = [
#     (0.0, 0.0),    # Inside
#     (0.5, 0.5),    # Inside
#     (2.0, 0.0),    # Outside
#     (1.0, 0.0),    # On boundary
# ]

# for x, y in test_points:
#     result = is_point_in_path_fourier(x, y, boundary_points)
#     print(f"Point ({x}, {y}): {result}")