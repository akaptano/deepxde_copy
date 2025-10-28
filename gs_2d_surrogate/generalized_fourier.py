import numpy as np
from deepxde.geometry.geometry_nd import GeometryND

class GeneralizedHyperEllipticalToroid(GeometryND):
    def __init__(
        self,
        eps_range=(0.1, 0.3),
        kappa_range=(0.1, 0.3),
        delta_range=(0.1, 0.3),
        x_ellipse=[],
        Amax=0.1,
        num_param=2,
        num_loops=4,  # Number of nested loops
        mpol=2,  # Number of Fourier modes
    ):
        self.N = 100
        self.num_param = num_param
        self.num_loops = num_loops
        self.mpol = mpol
        
        # Initialize center and parameter ranges
        self.center = np.array(
            [[0.0, 0.0, 0.0,
              eps_range[1] - eps_range[0],
              kappa_range[1] - kappa_range[0],
              delta_range[1] - delta_range[0]
              ]]
        )
        
        # Create parameter arrays
        self.tau = np.linspace(0, 2 * np.pi, self.N)
        Arange = np.linspace(-Amax, Amax, self.num_param)
        self.eps = np.linspace(eps_range[0], eps_range[1], self.num_param)
        self.kappa = np.linspace(kappa_range[0], kappa_range[1], self.num_param)
        self.delta = np.linspace(delta_range[0], delta_range[1], self.num_param)

        # Create arrays with dynamic shape based on num_loops
        shape = (self.N,) + (self.num_param,) * num_loops
        
        # Initialize base arrays for Fourier coefficients
        R0 = np.ones(shape)  # R0 coefficient is 1 for circular base shape
        Z0 = np.zeros(shape) # Z0 coefficient is 0 for circular base shape

        # Initialize arrays for higher order Fourier coefficients
        Rm_coeffs = []
        Zm_coeffs = []
        
        # For each Fourier mode m, create coefficient arrays
        for m in range(1, mpol + 1):
            # Create coefficient arrays with proper shape
            Rm_m = np.ones(shape)
            Zm_m = np.ones(shape)
            Rm_coeffs.append(Rm_m)
            Zm_coeffs.append(Zm_m)

        # Stack all coefficients together
        Rm_grid = np.stack([R0] + Rm_coeffs, axis=-1)
        Zm_grid = np.stack([Z0] + Zm_coeffs, axis=-1)

        # Create arrays for the final shape
        R_ellipse = np.zeros(shape)
        Z_ellipse = np.zeros(shape)
        A_ellipse = np.zeros(shape)
        eps_ellipse = np.zeros(shape)
        kappa_ellipse = np.zeros(shape)
        delta_ellipse = np.zeros(shape)

        # Create indices for all combinations
        indices = np.indices((self.num_param,) * num_loops).reshape(num_loops, -1).T

        # Fill arrays using vectorized operations
        for idx in indices:
            # Create slice for current combination
            slc = (slice(None),) + tuple(idx)
            
            # Calculate values using Fourier coefficients
            R_ellipse[slc] = 1 + self.eps[idx[1]] * np.cos(self.tau + np.arcsin(self.delta[idx[2]]) * np.sin(self.tau))
            Z_ellipse[slc] = self.eps[idx[1]] * self.kappa[idx[2]] * np.sin(self.tau)
            A_ellipse[slc] = Arange[idx[0]]
            eps_ellipse[slc] = self.eps[idx[1]]
            kappa_ellipse[slc] = self.kappa[idx[2]]
            delta_ellipse[slc] = self.delta[idx[2]]

        # Define boundary of hyper-elliptical disk
        self.x_ellipse = np.transpose(
            np.asarray(
                [R_ellipse, Z_ellipse, A_ellipse,
                 eps_ellipse, kappa_ellipse, delta_ellipse]),
            list(range(1, num_loops + 2)) + [0]
        )
        
        # Reshape to final form
        self.x_ellipse = self.x_ellipse.reshape(self.N * self.num_param ** num_loops, 6)

        # Set bounding box
        xmin = np.array([1 - np.max(self.eps), -np.max(self.kappa * self.eps), -Amax, 
                        eps_range[0], kappa_range[0], delta_range[0]])
        xmax = np.array([1 + np.max(self.eps), np.max(self.kappa * self.eps), Amax, 
                        eps_range[-1], kappa_range[-1], delta_range[-1]])
        self.Amax = Amax

        super(GeneralizedHyperEllipticalToroid, self).__init__(6, (xmin, xmax), 1)

    def inside(self, x):
        return is_point_in_path(x[:, 0:1], x[:, 1:2], self.x_ellipse)

    def on_boundary(self, x):
        return np.array([self.point_on_boundary(x[i]) for i in range(len(x))])

    def point_on_boundary(self, x):
        tol = np.max(np.linalg.norm(self.x_ellipse[:-1, 0:2] - self.x_ellipse[1:, 0:2], axis=-1))
        abs_diff = np.abs(x[:, 0:2] - self.x_ellipse[:, 0:2])
        return np.any(np.sqrt(abs_diff[:, 0:1]**2 + abs_diff[:, 1:2]**2) <= tol) 