import numpy as np
import deepxde as dde

class ToroidalGeometry(dde.geometry.geometry_nd.Geometry):
    """A custom geometry class for toroidal plasma shapes."""
    
    def __init__(self, eps0, kappa0, delta0, Amax):
        super().__init__(2, (eps0[0], kappa0[0], delta0[0], -Amax), (eps0[1], kappa0[1], delta0[1], Amax))
        self.eps0 = eps0
        self.kappa0 = kappa0
        self.delta0 = delta0
        self.Amax = Amax

    def inside(self, x):
        """Check if a point is inside the toroidal domain."""
        eps, kappa, delta, A = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        
        # Check if parameters are within their ranges
        eps_in_range = (eps >= self.eps0[0]) & (eps <= self.eps0[1])
        kappa_in_range = (kappa >= self.kappa0[0]) & (kappa <= self.kappa0[1])
        delta_in_range = (delta >= self.delta0[0]) & (delta <= self.delta0[1])
        A_in_range = (A >= -self.Amax) & (A <= self.Amax)
        
        return eps_in_range & kappa_in_range & delta_in_range & A_in_range

    def on_boundary(self, x):
        """Check if a point is on the boundary of the toroidal domain."""
        eps, kappa, delta, A = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        
        # Check if any parameter is at its boundary
        eps_on_boundary = np.isclose(eps, self.eps0[0]) | np.isclose(eps, self.eps0[1])
        kappa_on_boundary = np.isclose(kappa, self.kappa0[0]) | np.isclose(kappa, self.kappa0[1])
        delta_on_boundary = np.isclose(delta, self.delta0[0]) | np.isclose(delta, self.delta0[1])
        A_on_boundary = np.isclose(A, -self.Amax) | np.isclose(A, self.Amax)
        
        return eps_on_boundary | kappa_on_boundary | delta_on_boundary | A_on_boundary

    def random_points(self, n, random="pseudo"):
        """Generate random points inside the domain."""
        eps = np.random.uniform(self.eps0[0], self.eps0[1], (n, 1))
        kappa = np.random.uniform(self.kappa0[0], self.kappa0[1], (n, 1))
        delta = np.random.uniform(self.delta0[0], self.delta0[1], (n, 1))
        A = np.random.uniform(-self.Amax, self.Amax, (n, 1))
        
        return np.hstack((eps, kappa, delta, A)) 