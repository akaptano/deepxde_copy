from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import deepxde as dde
import os
from deepxde.backend import tf

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

x,u = gen_traindata(8)

n_test = 8
x_test,u_test = gen_traindata(n_test)
x_test = np.concatenate((x_test, spatial_domain.random_points(n_test)))
u_test = np.concatenate((u_test, np.zeros((n_test, 1))))

# specify psi, psi_r, psi_z, psi_rr, psi_zz at four locations 

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

# x[0] = R, x[1] = Z, u[0] = U
bc2 = dde.OperatorBC(spatial_domain ,
                    lambda x, u, _: psi_zz(x, u)+ N1*psi_r(x,u), 
                    boundary_outer)

bc4 = dde.OperatorBC(spatial_domain,
                    lambda x, u, _: psi_zz(x, u)+ N2*psi_r(x,u), 
                    boundary_inner)

bc6 = dde.OperatorBC(spatial_domain ,lambda x, u, _: psi_r(x, u), boundary_high)
bc7 = dde.OperatorBC(spatial_domain ,lambda x, u, _: psi_rr(x, u)+ N3 * psi_z(x, u), boundary_high)


data = dde.data.PDE(
    spatial_domain,
    pde_solovev,
    [bc135],
    #anchors=observe_x,
    num_domain=8,
    num_boundary=0,
    num_test=n_test,
    train_distribution="LHS"
)

from utils.gs_solovev_sol import GS_Linear
ITER = GS_Linear(eps= 0.32, kappa=1.7, delta=0.33)
ITER.get_BCs(A=-0.155)
ITER.solve_coefficients()

LR = 2e-1
DEPTH = 1
BREADTH = 5
AF = 'swish'
net = dde.maps.FNN([2] + DEPTH * [BREADTH] + [1], AF, "Glorot normal")

model = dde.Model(data, net)


# Compile, train and save model
model.compile(
    "L-BFGS-B",
    loss_weights=[1,1]
)
loss_history, train_state = model.train(epochs=1000, display_every = 100)
#dde.saveplot(loss_history, train_state, save_plot=True,issave=True, isplot=True,output_dir=f'./cefron/{CONFIG}/runs/{RUN_NAME}')

# Evaluation
from utils.utils import evaluate,evaluate_eq, relative_error_plot
x,y,psi_pred,psi_true,error=evaluate(ITER,model)

# make mesh
print(np.mean(error) * 100)
print(np.max(error) * 100)


# Store Values
engineering_params = {
    "true_volume": 0.0,
    "pred_volume": 0.0,
    "true_Cp": 0.0,
    "pred_Cp": 0.0,
    "true_qstar": 0.0,
    "pred_qstar": 0.0,
    "true_beta_p": 0.0,
    "pred_beta_p": 0.0,
    "true_beta_t": 0.0,
    "pred_beta_t": 0.0,
    "true_beta": 0.0,
    "pred_beta": 0.0,
}

# Compute a contour integral
def area(vs):
    a = 0
    x0, y0 = vs[0]
    for [x1, y1] in vs[1:]:
        dx = x1 - x0
        dy = y1 - y0
        a += 0.5 * (y0 * dx - x0 * dy)
        x0 = x1
        y0 = y1
    return a

# Compute Volume from psi = 0 flux surface for true and predicted
c = plt.contour(x, y, psi_true, [0])
v = c.collections[0].get_paths()[0].vertices
print('True volume = ', area(v))
engineering_params["true_volume"] = area(v)

c = plt.contour(x, y, psi_pred, [0])
v = c.collections[0].get_paths()[0].vertices
print('predicted volume = ', area(v))
engineering_params["pred_volume"] = area(v)

def Cp(vs):
    a = 0
    x0, y0 = vs[0]
    for [x1, y1] in vs[1:]:
        dx = x1 - x0
        dy = y1 - y0
        dy_dx = dy / dx
        a += np.sqrt(1 + dy_dx ** 2) * abs(dx)
        x0 = x1
        y0 = y1
    return a

# Compute Cp from psi = 0 flux surface for true and predicted
c = plt.contour(x, y, psi_true, [0])
v = c.collections[0].get_paths()[0].vertices
print('True Cp = ', Cp(v))
engineering_params["true_Cp"] = Cp(v)

c = plt.contour(x, y, psi_pred, [0])
v = c.collections[0].get_paths()[0].vertices
print('predicted Cp = ', Cp(v))
engineering_params["pred_Cp"] = Cp(v)

# Using Green's theorem again
def qstar_integral(vs):
    a = 0
    x0, y0 = vs[0]
    for [x1, y1] in vs[1:]:
        dx = x1 - x0
        dy = y1 - y0
        M = - 1 / x1
        a += M * dy
        x0 = x1
        y0 = y1
    return a

# Compute qstar from psi = 0 flux surface for true and predicted
c = plt.contour(x, y, psi_true, [0])
v = c.collections[0].get_paths()[0].vertices
mu0 = 4 * np.pi * 10 ** (-7)
I = 15 * 10 ** 6
a = 2.0
R0 = 6.2
epsilon = 0.32
B0 = 5.3
psi0 = - mu0 * I * a / epsilon / (-0.155 * qstar_integral(v) + 1.115 * area(v))
qstar = - (a * R0 * B0 * Cp(v)) / (psi0 * (-0.155 * qstar_integral(v) + 1.115 * area(v)))
print('True qstar = ', qstar)
engineering_params["true_qstar"] = qstar

psi_average = np.trapz(np.trapz(psi_true * x[0, :], x[0, :], axis=0), y[:, 0])
beta_p = 2 * 1.155 * Cp(v) ** 2 * psi_average / (
    area(v) * (-0.155 * qstar_integral(v) + 1.115 * area(v)) ** 2
)
print('True beta_p = ', beta_p)
print('True beta_t = ', epsilon ** 2 * beta_p / qstar ** 2)
print('True beta = ', epsilon ** 2 * beta_p / (qstar ** 2 + epsilon ** 2))
engineering_params["true_beta_p"] = beta_p
engineering_params["true_beta_t"] = epsilon ** 2 * beta_p / qstar ** 2
engineering_params["true_beta"] = epsilon ** 2 * beta_p / (qstar ** 2 + epsilon ** 2)


c = plt.contour(x, y, psi_pred, [0])
v = c.collections[0].get_paths()[0].vertices
psi_average = np.trapz(np.trapz(psi_pred * x[0, :], x[0, :], axis=0), y[:, 0])
psi0 = - mu0 * I * a / epsilon / (-0.155 * qstar_integral(v) + 1.115 * area(v))
qstar = - (a * R0 * B0 * Cp(v)) / (psi0 * (-0.155 * qstar_integral(v) + 1.115 * area(v)))
print('Predicted qstar = ', qstar)
engineering_params["pred_qstar"] = qstar

beta_p = 2 * 1.155 * Cp(v) ** 2 * 0.018170271593863394 / (
    area(v) * (-0.155 * qstar_integral(v) + 1.115 * area(v)) ** 2
)
print('Predicted beta_p = ', beta_p)
print('Predicted beta_t = ', epsilon ** 2 * beta_p / qstar ** 2)
print('Predicted beta = ', epsilon ** 2 * beta_p / (qstar ** 2 + epsilon ** 2))
engineering_params["pred_beta_p"] = beta_p
engineering_params["pred_beta_t"] = epsilon ** 2 * beta_p / qstar ** 2
engineering_params["pred_beta"] = epsilon ** 2 * beta_p / (qstar ** 2 + epsilon ** 2)

print('Normalized percent error in volume = ', abs(engineering_params['true_volume'] - engineering_params['pred_volume']) / engineering_params['true_volume'])
print('Normalized percent error in Cp = ', abs(engineering_params['true_Cp'] - engineering_params['pred_Cp']) / engineering_params['true_Cp'])
print('Normalized percent error in qstar = ', abs(engineering_params['true_qstar'] - engineering_params['pred_qstar']) / engineering_params['true_qstar'])
print('Normalized percent error in beta_p = ', abs(engineering_params['true_beta_p'] - engineering_params['pred_beta_p']) / engineering_params['true_beta_p'])

# Evaluation
# make mesh
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
