from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
from matplotlib import pyplot as plt
import deepxde as dde
from scipy.special import jn_zeros, jv
A = 1
mu0 = 1
psi0 = 0.1
p0 = 0.1
k = np.pi


def pde_0beta(x, u):
    lamda = np.sqrt(jn_zeros(1, 1) ** 2 + np.pi ** 2) 
    gamma = np.sqrt(lamda ** 2 - k ** 2)
    psi = u[:, 0:1]
    psi_r = dde.grad.jacobian(u, x, i=0, j=0)
    psi_rr = dde.grad.hessian(u, x, i=0, j=0)
    psi_zz = dde.grad.hessian(u, x, i=1, j=1)
    GS = (
         psi_rr - psi_r / x[:, 0:1] + psi_zz + lamda ** 2 * psi
    )

    return [GS]


def pde_linear(x, u):
    lamda = np.sqrt(np.pi ** 2 - 2 ** 2) 
    gamma = np.sqrt(- lamda ** 2 + k ** 2)
    psi = u[:, 0:1]
    psi_r = dde.grad.jacobian(u, x, i=0, j=0)
    psi_rr = dde.grad.hessian(u, x, i=0, j=0)
    psi_zz = dde.grad.hessian(u, x, i=1, j=1)
    GS = (
        psi_rr - psi_r / x[:, 0:1] + psi_zz + x[:, 0:1] ** 2 + lamda ** 2 * psi
    )
    return [GS]


def psi_func(x):
    return 0.0
    

def psi_0beta_analytic(x):
    lamda = np.sqrt(jn_zeros(1, 1) ** 2 + np.pi ** 2) 
    gamma = np.sqrt(lamda ** 2 - k ** 2)
    r0 = jn_zeros(0, 1) / gamma
    return psi0 * x[:, 0:1] * jv(1, gamma * x[:, 0:1]) * np.sin(k * x[:, 1:2]) / jv(1, gamma * r0) / r0
    

def psi_linear_analytic(x):
    lamda = np.sqrt(np.pi ** 2 - 2 ** 2) 
    gamma = np.sqrt(- lamda ** 2 + k ** 2)
    r0 = 1.652545 / gamma
    beta_fac = (1 + jv(0, gamma * r0) / jv(2, gamma * r0)) / (jv(1, gamma * r0) * r0)
    beta_term = x[:, 0:1] ** 2 * jv(0, gamma * r0) / (jv(2, gamma * r0) * r0 ** 2)
    return psi0 * (beta_fac * x[:, 0:1] * jv(1, gamma * x[:, 0:1]) * np.sin(k * x[:, 1:2]) - beta_term)


def main(argv):
    linear = False
    print(str(sys.argv))
    if np.any(np.asarray(str(sys.argv)) == 'linear'):
        linear = True

    spatial_domain = dde.geometry.Rectangle(xmin=[1e-4, 0.0], xmax=[1, 1])

    boundary_condition_psi = dde.DirichletBC(
        spatial_domain, lambda x: 0, lambda _, on_boundary: on_boundary
    )

    # Adding a single observation point in the center to make sure psi > 0 
    observe_x = np.asarray([0.5, 0.5]).reshape(1, 2)
    if linear:
        observe_y = dde.PointSetBC(observe_x, psi_linear_analytic(observe_x))
        pde = pde_linear
        soln = psi_linear_analytic
    else:
        observe_y = dde.PointSetBC(observe_x, psi_0beta_analytic(observe_x))
        pde = pde_0beta
        soln = psi_0beta_analytic 

    data = dde.data.PDE(
        spatial_domain,
        pde,
        [boundary_condition_psi, observe_y],
        solution=soln,
        anchors=observe_x,
        num_domain=500,
        num_boundary=50,
        num_test=100,
    )

    net = dde.maps.FNN([2] + 4 * [40] + [1], "tanh", "Glorot normal")
    net.apply_output_transform(lambda x, y: (x[:, 0:1] * (1 - x[:, 0:1])) * (x[:, 1:2] * (1 - x[:, 1:2])) * y)

    model = dde.Model(data, net)

    model.compile(
        "adam", lr=1e-3, metrics=["l2 relative error"]
    )
    model.train(epochs=500)
    model.compile("L-BFGS-B", metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=500)
    #dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    # make mesh
    nr = 50
    nz = 50
    r, z = np.meshgrid(
        np.linspace(1e-4, 1, nr),
        np.linspace(0.0, 1, nz),
    )

    X = np.vstack((np.ravel(r), np.ravel(z))).T

    output = model.predict(X)

    # psi is only predicted up to overall constant 
    # so normalize to psi0 
    psi_pred = output[:, 0].reshape(-1)
    psi_pred = psi_pred / np.max(np.abs(psi_pred)) * psi0
    psi_pred = np.reshape(psi_pred, [nr, nz])
    print(psi_pred.shape)
    print(X.shape, r.shape)
    plt.figure(figsize=(10, 14))
    plt.subplot(3, 1, 1)
    plt.contourf(r, z, psi_pred)
    plt.colorbar()  # ticks=np.linspace(0, 0.105, 10))
    plt.ylabel('Z', fontsize=24)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xticklabels([])

    if linear:
        psi_true = psi_linear_analytic(X)
    else:    
        psi_true = psi_0beta_analytic(X)
    psi_true = np.reshape(psi_true, [nr, nz])
    plt.subplot(3, 1, 2)
    plt.contourf(r, z, psi_true)
    plt.colorbar()  # ticks=np.linspace(0, 0.105, 10))
    plt.ylabel('Z', fontsize=24)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xticklabels([])
    
    plt.subplot(3, 1, 3)
    plt.contourf(r, z, np.abs(psi_true - psi_pred))
    plt.colorbar()
    plt.xlabel('R', fontsize=24)
    plt.ylabel('Z', fontsize=24)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    plt.savefig('psi_PINN.jpg')
    # psi_exact = psi_func(X).reshape(-1)

    GS = model.predict(X, operator=pde)

    residual_psi = np.mean(np.absolute(GS))

    print("Accuracy")
    print("Mean residual:", residual_psi)
    # print("L2 relative error in u:", l2_difference_psi)

if __name__ == "__main__":
    main(sys.argv)
