from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import deepxde as dde

def create_model():
    A = -0.155
    eps = 0.32
    kappa = 1.7
    delta = 0.33

    N1 = - (1 + np.arcsin(delta)) ** 2 / (eps * kappa ** 2)
    N2 = (1 - np.arcsin(delta)) ** 2 / (eps * kappa ** 2)
    N3 = - kappa / (eps * np.cos(np.arcsin(delta)) ** 2)

    def gen_traindata(num):
        eps = 0.32
        kappa = 1.7
        delta = 0.33
        N = num
        center, eps, kappa, delta = np.array([[0.0,0.0]]), eps, kappa, delta
        tau = np.linspace(0, 2 * np.pi, N)
        # Define boundary of ellipse
        x_ellipse = np.asarray([1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau)), 
                        eps * kappa * np.sin(tau)]).T
        xvals = x_ellipse
        uvals = np.zeros(len(xvals)).reshape(len(xvals), 1)
        return xvals, uvals


    def pde_solovev(x, u):
        psi = u[:, 0:1]
        psi_r = dde.grad.jacobian(psi, x, i=0, j=0)
        psi_rr = dde.grad.hessian(psi, x, i=0, j=0)
        psi_zz = dde.grad.hessian(psi, x, i=1, j=1)
        GS = psi_rr - psi_r / x[:, 0:1] + psi_zz - (1 - A) * x[:, 0:1] ** 2 - A
        return GS

    def psi_r(x,u):
        return dde.grad.jacobian(u, x, i=0, j=0)
    def psi_z(x,u):
        return  dde.grad.jacobian(u, x, i=0, j=1)
    def psi_rr(x, u):
        return dde.grad.hessian(u, x, i=0, j=0)
    def psi_zz(x, u):
        return dde.grad.hessian(u, x, i=1, j=1)

    def boundary_outer(x, on_boundary):
        return on_boundary and np.isclose([x[0], x[1]], [1 + eps, 0]).all()
    def boundary_inner(x, on_boundary):
        return on_boundary and np.isclose([x[0], x[1]], [1 - eps, 0]).all()
    def boundary_high(x, on_boundary):
        return on_boundary and np.isclose([x[0], x[1]], [1 - delta * eps, kappa * eps]).all()

    spatial_domain = dde.geometry.Ellipse(eps, kappa, delta) 

    x,u = gen_traindata(1000)

    # specify psi, psi_r, psi_z, psi_rr, psi_zz at four locations 
    # observe_x = np.asarray([[1 + eps, 0], [1 - eps, 0], [1 - delta * eps, kappa * eps]])
    # observe_y = np.asarray([0.0, 0.0, 0.0]).reshape(3, 1)
    observe_x = np.asarray([[1 + eps, 0], 
                            [1 - eps, 0], 
                            [1 - delta * eps, kappa * eps],
                            [1 - delta * eps, -kappa * eps]]
                        )
    observe_y = np.asarray([0.0, 0.0, 0.0,0.0]).reshape(4, 1)

    observe_x = np.concatenate((x,observe_x))
    observe_y = np.concatenate((u,observe_y))

    bc135 = dde.PointSetBC(x,u)
    # bc135 = dde.PointSetBC(observe_x, observe_y)

    # x[0] = R
    # x[1] = Z
    # u[0] = U
    # is x all the boundary points?
    bc2 = dde.OperatorBC(spatial_domain ,
                        lambda x, u, _: psi_zz(x, u)+ N1*psi_r(x,u), 
                        boundary_outer)

    bc4 = dde.OperatorBC(spatial_domain,
                        lambda x, u, _: psi_zz(x, u)+ N2*psi_r(x,u), 
                        boundary_inner)

    bc6 = dde.OperatorBC(spatial_domain ,lambda x, u, _: psi_r(x, u), boundary_high)
    bc7 = dde.OperatorBC(spatial_domain ,lambda x, u, _: psi_rr(x, u)+N3*psi_z(x,u), boundary_high)


    data = dde.data.PDE(
        spatial_domain,
        pde_solovev,
        [bc135],
        # [bc135, bc2, bc4,bc6,bc7],
        anchors=observe_x,
        num_domain=1024,
        num_boundary=0,
        num_test=100,
        train_distribution="LHS"
    )

    net = dde.maps.FNN([2] + 4 * [40] + [1], "tanh", "Glorot normal")

    model = dde.Model(data, net)
    return model