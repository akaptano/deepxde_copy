from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import deepxde as dde
from scipy.special import jn_zeros, jv

def main():
    A = -0.155
    eps = 0.32
    kappa = 1.7
    delta = 0.33
    B0 = 5.3 
    I0 = 15e6
    Cp = 2.79
    Vnorm = 0.53
    qstar = 1.57
    betat = 0.05

    def pde_solovev(x, u):
        psi = u[:, 0:1]
        psi_r = dde.grad.jacobian(u, x, i=0, j=0)
        psi_rr = dde.grad.hessian(u, x, i=0, j=0)
        psi_zz = dde.grad.hessian(u, x, i=1, j=1)
        GS = (
            psi_rr - psi_r / x[:, 0:1] + psi_zz - (1 - A) * x[:, 0:1] ** 2 - A
        )

        return [GS]

    def psi_func(x):
        return 0.0
    
    def psi_solovev_analytic(x):
        psi_particular = (1 - A) * x[:, 0:1] ** 4 / 8.0 + A * x[:, 0:1] ** 2 * np.log(x[:, 0:1]) / 2.0
        # Define the first 7 polynomials (up to 6th order) -- see Friedbergs Ideal MHD
        U0 = 1
        U1 = x[:, 0:1] ** 2
        U2 = x[:, 1:2] ** 2 - x[:, 0:1] ** 2 * np.log(x[:, 0:1])
        U3 = x[:, 0:1] ** 4 - 4.0 * x[:, 0:1] ** 2 * x[:, 1:2] ** 2
        U4 = 2.0 * x[:, 1:2] ** 4 - 9.0 * x[:, 0:1] ** 2 * x[:, 1:2] ** 2 - (12.0 * x[:, 0:1] ** 2 * x[:, 1:2] ** 2 - 3 * x[:, 0:1] ** 4) * np.log(x[:, 0:1]) 
        U5 = x[:, 0:1] ** 6 - 12.0 * x[:, 0:1] ** 4 * x[:, 1:2] ** 2 + 8.0 * x[:, 0:1] ** 2 * x[:, 1:2] ** 4
        U6 = 8.0 * x[:, 1:2] ** 6 - 14.0 * x[:, 0:1] ** 2 * x[:, 1:2] ** 4 + 75.0 * x[:, 0:1] ** 4 * x[:, 1:2] ** 2 - (120.0 * x[:, 0:1] ** 2 * x[:, 1:2] ** 4 - 180 * x[:, 0:1] ** 4 * x[:, 1:2] ** 2 + 15.0 * x[:, 0:1] ** 6) * np.log(x[:, 0:1]) 
        # solve for the 6 coefficients 
        
        psi_homogeneous = 0.0  # todo 
        return psi_particular + psi_homogeneous 

    spatial_domain = dde.geometry.Ellipse(eps, kappa, delta) 

    # specify psi, psi_r, psi_z, psi_rr, psi_zz at three locations 
    observe_x = np.asarray([[1 + eps, 0], [1 - eps, 0], [1 - delta * eps, kappa * eps]])
    psi_outerEquatorial = 0.0
    psi_innerEquatorial = 0.0
    psi_highPoint = 0.0
    dpsi_dr_highPoint = 0.0
    N1 = - (1 + np.arcsin(delta)) ** 2 / (eps * kappa ** 2)
    N2 = (1 - np.arcsin(delta)) ** 2 / (eps * kappa ** 2)
    N3 = - kappa / (eps * np.cos(np.arcsin(delta)) ** 2)
    dpsi_dr_outerEquatorial = 0.0  # todo
    dpsi_dr_innerEquatorial = 0.0  # todo
    dpsi_dz_highPoint = 0.0  # todo
    d2psi_d2z_outerEquatorial = -N1 * dpsi_dr_outerEquatorial
    d2psi_d2z_innerEquatorial = -N2 * dpsi_dr_innerEquatorial
    d2psi_d2r_highPoint = -N3 * dpsi_dz_highPoint
    observe_y = np.asarray([psi_outerEquatorial, psi_innerEquatorial, psi_highPoint]).reshape(3, 1)
    observe_y = dde.PointSetBC(observe_x, observe_y)
    data = dde.data.PDE(
        spatial_domain,
        pde_solovev,
        observe_y,
        #[observe_y],
        solution=psi_solovev_analytic,
        anchors=observe_x,
        num_domain=500,
        num_boundary=50,
        num_test=100,
    )

    net = dde.maps.FNN([2] + 4 * [40] + [1], "tanh", "Glorot normal")

    model = dde.Model(data, net)

    model.compile(
        "adam", lr=1e-3, metrics=["l2 relative error"]
    )
    model.train(epochs=500)
    model.compile("L-BFGS-B", metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=500)
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    # make mesh
    nx = 100
    ny = 100
    tau = np.linspace(0, 2 * np.pi, 1000)
    x, y = np.meshgrid(
        np.linspace(1 - 2 * eps, 1 + 2 * eps, nx),
        np.linspace(-kappa * eps, kappa * eps, ny),
    )

    X = np.vstack((np.ravel(x), np.ravel(y))).T

    output = model.predict(X)

    # psi is only predicted up to overall constant 
    # so normalize to psi0 
    psi_pred = output[:, 0].reshape(-1)
    # psi_pred = psi_pred / np.max(np.abs(psi_pred)) * psi0
    psi_pred = np.reshape(psi_pred, [nx, ny])
    print(psi_pred.shape)
    plt.figure(figsize=(10, 14))
    plt.subplot(3, 1, 1)
    plt.contourf(x, y, psi_pred)
    plt.colorbar()  # ticks=np.linspace(0, 0.105, 10))
    plt.ylabel(r'$Z/R_0$', fontsize=24)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xticklabels([])

    psi_true = psi_solovev_analytic(X)
    psi_true = np.reshape(psi_true, [nx, ny])
    plt.subplot(3, 1, 2)
    plt.contourf(x, y, psi_true)
    plt.colorbar()  # ticks=np.linspace(0, 0.105, 10))
    plt.ylabel(r'$Z/R_0$', fontsize=24)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xticklabels([])
    
    plt.subplot(3, 1, 3)
    plt.contourf(x, y, np.abs(psi_true - psi_pred))
    plt.colorbar()
    plt.xlabel(r'$R/R_0$', fontsize=24)
    plt.ylabel(r'$Z/R_0$', fontsize=24)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.savefig('PINN_solovev.jpg')
    # psi_exact = psi_func(X).reshape(-1)

    GS = model.predict(X, operator=pde_solovev)

    residual_psi = np.mean(np.absolute(GS))

    print("Accuracy")
    print("Mean residual:", residual_psi)
    # print("L2 relative error in u:", l2_difference_psi)

if __name__ == "__main__":
    main()
