from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import deepxde as dde
import os
from utils.gs_solovev_sol import GS_Linear

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

######################
# ITER Configuration #
######################
A = -0.155
eps = 0.32
kappa = 1.7
delta = 0.33

N1 = - (1 + np.arcsin(delta)) ** 2 / (eps * kappa ** 2)
N2 = (1 - np.arcsin(delta)) ** 2 / (eps * kappa ** 2)
N3 = - kappa / (eps * np.cos(np.arcsin(delta)) ** 2)

def gen_traindata(num):
    ######################
    # ITER Configuration #
    ######################
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

n_test = 100

x, u = gen_traindata(1024)
bc135 = dde.PointSetBC(x, u)
ITER = GS_Linear(eps=0.32, kappa=1.7, delta=0.33)
ITER.get_BCs(A=-0.155)
ITER.solve_coefficients()
colors = ['r', 'b', 'g', 'k', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'lime']
loss_ratio = np.logspace(-5, 5, 11, endpoint=True)

errors = np.zeros((11, 4, 501 * 501))
for i in range(11):
    for j in range(4):
        data = dde.data.PDE(
            spatial_domain,
            pde_solovev,
            [bc135],
            num_domain=1024,
            num_boundary=0,
            num_test=n_test,
            train_distribution="LHS"
        )

        LR = 2e-2
        DEPTH = 2
        BREADTH = 40
        AF = 'swish'
        net = dde.maps.FNN([2] + DEPTH * [BREADTH] + [1], AF, "Glorot normal")
        model = dde.Model(data, net)

        # Compile, train and save model
        model.compile(
            "L-BFGS-B",
            loss_weights=[1, loss_ratio[i]]
        )
        losshistory, train_state = model.train(epochs=10000, display_every = 100)
        loss_train = np.sum(losshistory.loss_train, axis=1)
        loss_train_domain = [item[0] for item in losshistory.loss_train]
        loss_train_boundary = [item[1] for item in losshistory.loss_train]
        loss_test = np.sum(losshistory.loss_test, axis=1)

        if j == 0:
            plt.figure(1)
            plt.semilogy(losshistory.steps, loss_train_domain, color=colors[i], label="{0:.1e} domain train loss".format(loss_ratio[i]))
            plt.semilogy(losshistory.steps, loss_train_boundary / loss_ratio[i], color=colors[i], linestyle='--' )  # label=str(loss_ratio[i]) + " boundary train loss")
            # plt.semilogy(losshistory.steps, loss_test, label="Test loss")
            for i in range(len(losshistory.metrics_test[0])):
                plt.semilogy(
                    losshistory.steps,
                    np.array(losshistory.metrics_test)[:, i],
                    label="Test metric",
                )

        # Evaluation
        from utils.utils import evaluate,evaluate_eq, relative_error_plot
        x,y,psi_pred,psi_true,error=evaluate(ITER,model)
        errors[i, j, :] = np.sqrt(np.ravel(error))
        print(np.mean(error) * 100)
        print(np.max(error) * 100)

plt.xlabel("# Steps")
plt.legend()
plt.grid(True)

mean_errors = np.mean(np.mean(errors, axis=-1), axis=-1)
max_errors = np.mean(np.max(errors, axis=-1), axis=-1)
max_std_errors = np.std(np.max(errors, axis=-1), axis=-1)
std_errors = np.std(np.std(errors, axis=-1), axis=-1)
plt.figure(2)
print(loss_ratio.shape, mean_errors.shape, std_errors.shape, max_errors.shape, max_std_errors.shape)
plt.errorbar(loss_ratio, max_errors, max_std_errors,
             markeredgecolor='k', marker='^', label='Max errors')
plt.errorbar(loss_ratio, mean_errors, std_errors,
             markeredgecolor='k', marker='o', label='Mean errors')
plt.grid(True)
ax = plt.gca()
ax.set_xscale("log")
ax.set_yscale("log")
plt.legend()
plt.show()

# Evaluation
nx = 100
ny = 100
zoom = ((1 + eps)-(1 - eps))*0.05
innerPoint = 1 - eps - zoom
outerPoint = 1 + eps + zoom
lowPoint   = -kappa * eps - zoom
highPoint  = kappa * eps + zoom
x, y = np.meshgrid(
    np.linspace(innerPoint, outerPoint, nx),
    np.linspace(lowPoint, highPoint, ny),
)
X = np.vstack((np.ravel(x), np.ravel(y))).T
x,y,psi_pred,psi_true,error=evaluate(ITER,model)
x_eq, psi_true_eq, psi_pred_eq, e_eq= evaluate_eq(ITER,model)
X_test = spatial_domain.random_points(333)

# Plotting Setup
print(psi_pred.shape)
fig,axs=plt.subplots(2,2,figsize=(10,10))
ax1,ax2,ax3,ax4=axs[0][0],axs[0][1],axs[1][0],axs[1][1]
levels = np.linspace(min(psi_true.reshape(-1)),0,8)

# Plot 1 - PINN Solution
cp = ax1.contour(x, y, psi_pred,levels=levels)
# ax1.scatter(observe_x[:,0], observe_x[:,1], s = 2,c="black")
fig.colorbar(cp,ax=ax1).formatter.set_powerlimits((0, 0))
ax1.set_title('PINN Solution')
ax1.set_xlabel(r'$R/R_{0}$')
ax1.set_ylabel(r'$Z/R_{0}$')
ax1.axis(xmin=innerPoint,xmax=outerPoint,ymin=lowPoint, ymax=highPoint)

# Plot 2 - Analytic Solution
cp = ax2.contour(x, y, psi_true,levels=levels)
fig.colorbar(cp,ax=ax2).formatter.set_powerlimits((0, 0))
ax2.set_title('Analytical Solution')
ax2.set_xlabel(r'$R/R_{0}$')
ax2.set_ylabel(r'$Z/R_{0}$')
ax2.axis(xmin=innerPoint,xmax=outerPoint,ymin=lowPoint, ymax=highPoint)

# Plot 3 - Equatorial Error
twin3 = ax3.twinx()
ax3.plot(x_eq, -psi_pred_eq,marker="+",color="red",label="neural netowrk")
ax3.plot(x_eq, -psi_true_eq,color="blue",label="analytic")
twin3.plot(x_eq, e_eq, color='red',linestyle='--',label="error")
twin3.yaxis.get_major_formatter().set_scientific(True)
ax3.set_title('error in z=0')
ax3.set_xlabel('R/R_0')
ax3.set_ylabel(r'$\psi(r,z=0)$')
ax3.legend(loc='upper left')
twin3.legend(loc='upper right')
twin3.set_ylabel('error', color='red')
plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))

# Plot 4 - Relative Error
fig, ax4 = relative_error_plot(fig,ax4,x,y,error,model,ITER,X_test=X_test)
# ax4.set_title(r'$($\psi$_{n}-u^{*})^2/u_{a}^2$')
ax4.set_title(r'($\psi_{a}-\psi^{*})^2/\psi_{a}^2$')
ax4.set_xlabel(r'$R/R_{0}$')
ax4.set_ylabel(r'$Z/R_{0}$')
ax4.axis(xmin=innerPoint,xmax=outerPoint,ymin=lowPoint, ymax=highPoint)

fig.tight_layout()
plt.savefig('fast_solve_summary.png')

model.print_model()
plt.show()
