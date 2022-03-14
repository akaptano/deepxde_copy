import sys
import numpy as np
import h5py
import matplotlib
#matplotlib.use('pdf')
from matplotlib import pyplot as plt
import deepxde as dde
from scipy.special import jn_zeros, jv
import os
from scipy.interpolate import griddata
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
psi0 = 0.1
p0 = 5
k = np.pi
with h5py.File("Spheromak-flat_lam-flat_press/psi_gs-500.rst", 'r') as fid:
    r_plot = np.asarray(fid['mesh/r_plot'])
    psi_h5py = np.asarray(fid['gs/psi'])
r_plot[:, 1] = r_plot[:, 1] + 0.5


# Plot losses for training and testing
def plot_loss():
    loss = np.loadtxt('loss.dat')
    test = np.loadtxt('test.dat')
    train = np.loadtxt('train.dat')
    steps = loss[:, 0]
    plt.figure()
    plt.semilogy(steps, loss[:, 1], label='train')
    plt.semilogy(steps, loss[:, 4], label='test')
    plt.legend()
    plt.grid(True)
    # plt.savefig('loss.png')


# Zero-beta GS equation
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


# Linear-pressure GS equation
def pde_linear(x, u):
    lamda = 4.675191889330353
    psi = u[:, 0:1]
    psi_r = dde.grad.jacobian(u, x, i=0, j=0)
    psi_rr = dde.grad.hessian(u, x, i=0, j=0)
    psi_zz = dde.grad.hessian(u, x, i=1, j=1)
    GS = (
        psi_rr - psi_r / x[:, 0:1] + psi_zz + p0 * x[:, 0:1] ** 2 + lamda ** 2 * psi
    )
    return [GS]


# Zero-beta analytic solution
def psi_0beta_analytic(x):
    lamda = np.sqrt(jn_zeros(1, 1) ** 2 + np.pi ** 2) 
    gamma = np.sqrt(lamda ** 2 - k ** 2)
    r0 = jn_zeros(0, 1) / gamma
    psi = psi0 * x[:, 0:1] * jv(1, gamma * x[:, 0:1]) * np.sin(k * x[:, 1:2]) / jv(1, gamma * r0) / r0
    return psi
    

# Linear-pressure analytic solution (from PSI-Tri)
def psi_linear_analytic(x):
    psi = griddata(r_plot, psi_h5py, x, method='cubic')
    psi = np.reshape(psi, (np.shape(psi)[0], 1))
    return psi


# Main program
def main(argv):
    linear = False
    if len(argv) > 1: 
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
        data = dde.data.PDE(
            spatial_domain,
            pde,
            [boundary_condition_psi, observe_y], 
            solution=soln,
            anchors=observe_x,
            num_domain=5000,
            num_boundary=500,
            num_test=1000,
        )
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
            num_domain=5000,
            num_boundary=500,
            num_test=1000,
        )

    net = dde.maps.FNN([2] + 4 * [40] + [1], "tanh", "Glorot normal")
    net.apply_output_transform(lambda x, y: (x[:, 0:1] * (1 - x[:, 0:1])) * (x[:, 1:2] * (1 - x[:, 1:2])) * y)

    model = dde.Model(data, net)

    # decay_steps = 1000
    # decay_rate = 0.95
    model.compile(
        "adam", lr=1e-4, metrics=["l2 relative error"]
    )
    model.train(epochs=50000)
    model.compile("L-BFGS-B", metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=5000)
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)
    plot_loss() 
    # make mesh
    nr = 100
    nz = 100
    r, z = np.meshgrid(
        np.linspace(1e-4, 1, nr),
        np.linspace(0.0, 1, nz),
    )

    X = np.vstack((np.ravel(r), np.ravel(z))).T

    output = model.predict(X)

    # psi is only predicted up to overall constant 
    # if the pressure is zero, so normalize to psi0 in this case
    psi_pred = output[:, 0].reshape(-1)
    if linear:
        psi_pred = psi_pred / np.max(np.abs(psi_pred))
    else:
        psi_pred = psi_pred / np.max(np.abs(psi_pred)) * psi0
    psi_pred = np.reshape(psi_pred, [nr, nz])
    plt.figure(figsize=(10, 14))
    plt.subplot(3, 1, 1)
    plt.contourf(r, z, psi_pred)
    plt.colorbar(ticks=np.linspace(0, 1.05, 10))
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
    plt.colorbar(ticks=np.linspace(0, 1.05, 10))
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
    # plt.savefig('psi_PINN.png')
    GS = model.predict(X, operator=pde)
    residual_psi = np.mean(np.absolute(GS))

    print("Accuracy")
    print("Mean residual:", residual_psi)
    # print("L2 relative error in u:", l2_difference_psi)
    plt.show()

if __name__ == "__main__":
    main(sys.argv)
