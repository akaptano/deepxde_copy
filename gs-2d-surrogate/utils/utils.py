import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def evaluate_eq(ITER,model):
    # make mesh
    nx = 100
    zoom = 0.00
    # Get equatorial plane y = 0
    x_eq, y_eq = np.meshgrid(
        np.linspace(1 - ITER.eps*(1+zoom), 1 + ITER.eps*(1+zoom), nx),
        np.linspace(0,0, 1),
    )
    X_eq = np.vstack((np.ravel(x_eq), np.ravel(y_eq))).T
    # Calculate corresponding psi
    psi_eq = []
    for point in X_eq:
        psi_eq.append(ITER.psi_func(point[0],point[1]))
    psi_true_eq = np.reshape(psi_eq, [nx, 1])
    output_eq = model.predict(X_eq)
    psi_pred_eq = output_eq[:, 0].reshape(-1)
    psi_pred_eq = np.reshape(psi_pred_eq, [nx, 1])

    e = (psi_true_eq-psi_pred_eq)**2/min(psi_true_eq)**2
    print('Average normalized percent error on midplane = ', np.mean(np.sqrt(e)) * 100)
    print('Max normalized percent error on midplane = ', np.max(np.sqrt(e)) * 100)

    return x_eq.reshape(-1),psi_true_eq.reshape(-1), psi_pred_eq.reshape(-1), e


def evaluate(ITER,model):
    '''
    ##TODO: Need to break this file and refactor
    evaluate() create a meshgrid and calculate 
    psi_true,psi_pred, and relative error
    Input:
        ITER: GS_Linear function that contains shape parameter
        model: tf.model that contains the trained model
    output:
        psi_true: analytical solution
        psi_pred: evaluated value from PINN model
        error: relative error betwen psi_tre and psi_pred
    '''
    N = 1001
    eps, kappa, delta = ITER.eps, ITER.kappa, ITER.delta
    tau = np.linspace(0, 2 * np.pi, N)
    # Define boundary of ellipse
    x_ellipse = np.asarray([1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau)), 
                    eps * kappa * np.sin(tau)]).T[::-1]

    # # make mesh
    # nx = 500
    # ny = 500
    # zoom = 0.05
    # x, y = np.meshgrid(
    #     np.linspace(1 - eps*(1+zoom), 1 + eps*(1+zoom), nx),
    #     np.linspace(-kappa * eps*(1+zoom), kappa * eps*(1+zoom), ny),
    # )
    # X = np.vstack((np.ravel(x), np.ravel(y))).T

    # make mesh
    nx = 501
    ny = 501
    zoom = 0.2
    inner_point = (1 - 1.1*ITER.eps*(1+zoom))
    outer_point = (1 + 1.1*ITER.eps*(1+zoom))
    high_point  = (1.1*ITER.kappa * ITER.eps*(1+zoom) )
    low_point   = (-1.1*ITER.kappa * ITER.eps*(1+zoom) )
    x, y = np.meshgrid(
        np.linspace(inner_point, outer_point , nx),
        np.linspace(low_point, high_point, ny),
    )

    X = np.vstack((np.ravel(x), np.ravel(y))).T

    X_bc = np.vstack((np.ravel(x_ellipse[:,0]), np.ravel(x_ellipse[:,1]))).T
    output_bc_pred = model.predict(X_bc).reshape(-1)
    output_bc_true = []
    for point in X_bc:
        output_bc_true.append(ITER.psi_func(point[0],point[1]))

    # Calculate corresponding psi
    psi_true_lin = []
    for point in X:
        psi_true_lin.append(ITER.psi_func(point[0],point[1]))
    psi_true_lin = np.array(psi_true_lin)
    psi_true = np.copy(np.reshape(psi_true_lin, [nx, ny]))
    #filter

    psi_pred_lin = model.predict(X)
    psi_pred_lin = psi_pred_lin.reshape(-1)
    psi_pred = np.copy(np.reshape(psi_pred_lin, [nx, ny]))
    #filter


    #filter
    e_max = max((output_bc_pred-np.array(output_bc_true))**2/min(output_bc_pred)**2)

    e = (psi_pred_lin-np.array(psi_true_lin))**2/min(psi_pred_lin)**2
    error = np.reshape(e, [nx, ny])
    error[error>e_max] = e_max

    return x,y,psi_pred, psi_true, error

def relative_error_plot(fig,ax,x,y,error,model,ITER,DIVERTOR=False,v=[],X_test=[]):
    
    N = 1001
    eps, kappa, delta = ITER.eps, ITER.kappa, ITER.delta
    tau = np.linspace(0, 2 * np.pi, N)
    # Define boundary of ellipse
    x_ellipse = np.asarray([1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau)), 
                    eps * kappa * np.sin(tau)]).T[::-1]
    if DIVERTOR == True:
        x_ellipse = v
    X_bc = np.vstack((np.ravel(x_ellipse[:,0]), np.ravel(x_ellipse[:,1]))).T
    
    levels = 1000
    cmap= plt.cm.get_cmap("magma", levels+1)
    # Calculate corresponding psi
    if len(X_test) != 0:
        psi_test = []
        for point in X_test:
            psi_test.append(ITER.psi_func(point[0],point[1]))
        psi_true_test = np.reshape(psi_test, [len(psi_test), 1])
        output_test = model.predict(X_test)
        psi_pred_test = output_test[:, 0].reshape(-1)
        psi_pred_test = np.reshape(psi_pred_test, [len(psi_pred_test), 1])
        e = (psi_true_test-psi_pred_test)**2/min(psi_true_test)**2
        print('Average normalized percent error = ', np.mean(np.sqrt(e)) * 100)
        print('Max normalized percent error = ', np.max(np.sqrt(e)) * 100)
        levels = np.linspace(0.0,max(e)[0],levels+1)

    cp = ax.contourf(x, y, error,levels=levels, cmap=cmap)

    # Inside
    ax.scatter(model.data.train_x[:,0], model.data.train_x[:,1], s = 1.5, c="#0E0CB5")

    # Boundary
    ax.scatter(model.data.bc_points()[:,0], model.data.bc_points()[:,1], s = 2,c="#D12F24")
    cb = fig.colorbar(cp,ax=ax) # Add a colorbar to a plot
    cb.formatter.set_powerlimits((0, 0))
    cb.ax.yaxis.set_offset_position('right')
    cb.update_ticks()

    circ = patches.Polygon(xy=X_bc, transform=ax.transData)
    for coll in cp.collections:
        coll.set_clip_path(circ)

    return fig, ax
