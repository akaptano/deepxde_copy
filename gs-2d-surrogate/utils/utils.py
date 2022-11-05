import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm


def evaluate_eq(ITER, model):
    """
        Evaluate PINN solution error at the Z=0 midplane.
        Input:
            ITER: GS_Linear function that contains shape parameter
            model: tf.model that contains the trained model
        output:
            psi_true: analytical solution
            psi_pred: evaluated value from PINN model
            error: relative error betwen psi_tre and psi_pred
    """

    # make mesh
    nx = 100
    zoom = 0.00
    A, eps, kappa, delta = ITER.A, ITER.eps, ITER.kappa, ITER.delta

    # Get equatorial plane y = 0
    x_eq, y_eq = np.meshgrid(
        np.linspace(1 - eps * (1 + zoom), 1 + eps * (1 + zoom), nx),
        np.linspace(0, 0, 1),
    )
    ones = np.ones(nx)
    num_inputs = model.train_state.X_train.shape[-1]

    if num_inputs == 2:
        X_eq = np.vstack((np.ravel(x_eq), np.ravel(y_eq))).T
    elif num_inputs == 3:
        X_eq = np.vstack((
            np.ravel(x_eq), np.ravel(y_eq), A * ones,
        )).T
    elif num_inputs == 6:
        X_eq = np.vstack((
            np.ravel(x_eq), np.ravel(y_eq), A * ones,
            eps * ones, kappa * ones, delta * ones
        )).T

    # Calculate corresponding psi
    psi_eq = []
    for point in X_eq:
        psi_eq.append(ITER.psi_func(point[0], point[1]))
    psi_true_eq = np.reshape(psi_eq, [nx, 1])
    output_eq = model.predict(X_eq)
    psi_pred_eq = output_eq[:, 0].reshape(-1)
    psi_pred_eq = np.reshape(psi_pred_eq, [nx, 1])

    error = (psi_true_eq - psi_pred_eq) ** 2 / min(psi_true_eq) ** 2
    # print(
    #     'Average normalized percent error on midplane = ',
    #     np.mean(np.sqrt(error)) * 100
    # )
    # print(
    #     'Max normalized percent error on midplane = ',
    #     np.max(np.sqrt(error)) * 100
    # )
    return x_eq.reshape(-1), psi_true_eq.reshape(-1), psi_pred_eq.reshape(-1), error


def evaluate(ITER, model):
    '''
    Input:
        ITER: GS_Linear function that contains shape parameter
        model: tf.model that contains the trained model
    output:
        psi_true: analytical solution
        psi_pred: evaluated value from PINN model
        error: relative error betwen psi_tre and psi_pred
    '''
    N = 200
    A, eps, kappa, delta = ITER.A, ITER.eps, ITER.kappa, ITER.delta

    tau = np.linspace(0, 2 * np.pi, N)

    R_ellipse = 1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau))
    Z_ellipse = eps * kappa * np.sin(tau)
    x_ellipse = np.asarray([R_ellipse, Z_ellipse]).T

    # make mesh
    nx = 51
    ny = 51
    zoom = 0.2
    inner_point = (1 - 1.1 * eps * (1 + zoom))
    outer_point = (1 + 1.1 * eps * (1 + zoom))
    high_point = (1.1 * kappa * eps * (1 + zoom))
    low_point = (-1.1 * kappa * eps * (1 + zoom))
    x, y = np.meshgrid(
        np.linspace(inner_point, outer_point, nx),
        np.linspace(low_point, high_point, ny),
    )

    ones = np.ones(nx * ny)
    ones_ellipse = np.ones(N)

    num_inputs = model.train_state.X_train.shape[-1]
    if num_inputs == 2:
        X = np.vstack((
            np.ravel(x),
            np.ravel(y),
        )).T
        X_bc = np.vstack(
             (np.ravel(x_ellipse[:, 0]),
              np.ravel(x_ellipse[:, 1]),
              )
        ).T
    elif num_inputs == 3:
        X = np.vstack((
            np.ravel(x),
            np.ravel(y),
            A * ones,
        )).T
        X_bc = np.vstack(
             (np.ravel(x_ellipse[:, 0]),
              np.ravel(x_ellipse[:, 1]),
              A * ones_ellipse,
              )
        ).T
    elif num_inputs == 6:
        X = np.vstack((
            np.ravel(x), np.ravel(y), A * ones,
            eps * ones, kappa * ones, delta * ones
        )).T
        X_bc = np.vstack(
             (np.ravel(x_ellipse[:, 0]),
              np.ravel(x_ellipse[:, 1]),
              A * ones_ellipse,
              eps * ones_ellipse,
              kappa * ones_ellipse,
              delta * ones_ellipse
              )
        ).T

    output_bc_pred = model.predict(X_bc).reshape(-1)
    output_bc_true = []
    for point in X_bc:
        output_bc_true.append(ITER.psi_func(point[0], point[1]))

    # Calculate corresponding psi
    psi_true_lin = []
    for point in X:
        psi_true_lin.append(ITER.psi_func(point[0], point[1]))
    psi_true_lin = np.array(psi_true_lin)
    psi_true = np.copy(np.reshape(psi_true_lin, [nx, ny]))

    psi_pred_lin = model.predict(X)
    psi_pred_lin = psi_pred_lin.reshape(-1)
    psi_pred = np.copy(np.reshape(psi_pred_lin, [nx, ny]))

    e_max = max((output_bc_pred - np.array(output_bc_true)) ** 2 / min(output_bc_pred) ** 2)
    e = (psi_pred_lin - np.array(psi_true_lin)) ** 2 / min(psi_pred_lin) ** 2
    error = np.reshape(e, [nx, ny])
    error[error > e_max] = e_max
    return x, y, psi_pred, psi_true, error


def plot_summary_figure(ITER, model, X_test, PATH):
    """
        Make summary plots of the solution and normalized errors, as in
        the Kaltsas 2021 pinns for MHD paper.
    """

    x, y, psi_pred, psi_true, error = evaluate(ITER, model)
    x_eq, psi_true_eq, psi_pred_eq, e_eq = evaluate_eq(ITER, model)
    zoom = ((1 + ITER.eps) - (1 - ITER.eps)) * 0.05
    innerPoint = 1 - ITER.eps - zoom
    outerPoint = 1 + ITER.eps + zoom
    lowPoint = -ITER.kappa * ITER.eps - zoom
    highPoint = ITER.kappa * ITER.eps + zoom

    # Plotting Setup
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    ax1, ax2, ax3, ax4 = axs[0][0], axs[0][1], axs[1][0], axs[1][1]
    levels = np.linspace(min(psi_true.reshape(-1)), 0, 8)

    # Plot 1 - PINN Solution
    cp = ax1.contour(x, y, psi_pred, levels=levels)
    # ax1.scatter(observe_x[:,0], observe_x[:,1], s = 2,c="black")
    fig.colorbar(cp, ax=ax1).formatter.set_powerlimits((0, 0))
    ax1.set_title('PINN Solution')
    ax1.set_xlabel(r'$R/R_{0}$')
    ax1.set_ylabel(r'$Z/R_{0}$')
    ax1.axis(xmin=innerPoint, xmax=outerPoint, ymin=lowPoint, ymax=highPoint)

    # Plot 2 - Analytic Solution
    cp = ax2.contour(x, y, psi_true, levels=levels)
    fig.colorbar(cp, ax=ax2).formatter.set_powerlimits((0, 0))
    ax2.set_title('Analytical Solution')
    ax2.set_xlabel(r'$R/R_{0}$')
    ax2.set_ylabel(r'$Z/R_{0}$')
    ax2.axis(xmin=innerPoint, xmax=outerPoint, ymin=lowPoint, ymax=highPoint)

    # Plot 3 - Equatorial Error
    twin3 = ax3.twinx()
    ax3.plot(x_eq, -psi_pred_eq, marker="+", color="red", label="neural network")
    ax3.plot(x_eq, -psi_true_eq, color="blue", label="analytic")
    twin3.plot(x_eq, e_eq, color='red', linestyle='--', label="error")
    twin3.yaxis.get_major_formatter().set_scientific(True)
    ax3.set_title('error in z=0')
    ax3.set_xlabel('R/R_0')
    ax3.set_ylabel(r'$\psi(r,z=0)$')
    ax3.legend(loc='upper left')
    twin3.legend(loc='upper right')
    twin3.set_ylabel('error', color='red')
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

    # Plot 4 - Relative Error
    fig, ax4 = relative_error_plot(
        fig, ax4, x, y, error, model, ITER,
    )
    # ax4.set_title(r'$($\psi$_{n}-u^{*})^2/u_{a}^2$')
    ax4.set_title(r'($\psi_{a}-\psi^{*})^2/\psi_{a}^2$')
    ax4.set_xlabel(r'$R/R_{0}$')
    ax4.set_ylabel(r'$Z/R_{0}$')
    ax4.axis(xmin=innerPoint, xmax=outerPoint, ymin=lowPoint, ymax=highPoint)
    fig.tight_layout()
    plt.savefig(PATH + 'analysis_before_BFGS.jpg')


def compute_params(x, y, psi_true, psi_pred):
    """
        Compute a number of important physical and engineering
        quantities from the true and predicted flux function
        solutions.
        NOTE: assumes that the psi=0 contour is within the viewing
        frame of the 2D mesh in (x, y) for both the true and predicted
        flux function solutions.
    """

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

    # Note these are ITER parameters! Need to change for other configs
    Itor = 15 * 10 ** 6
    a = 2.0
    R0 = 6.2
    epsilon = 0.32
    B0 = 5.3
    psi0 = - mu0 * Itor * a / epsilon / (-0.155 * qstar_integral(v) + 1.115 * area(v))
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
    psi0 = - mu0 * Itor * a / epsilon / (-0.155 * qstar_integral(v) + 1.115 * area(v))
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
    return engineering_params


def relative_error_plot(
    fig, ax, x, y, error, model, ITER, DIVERTOR=False,
):
    """
        Make summary plot of the solution and normalized errors, as in
        the Kaltsas 2021 pinns for MHD paper.
    """

    N = 1001
    eps, kappa, delta = ITER.eps, ITER.kappa, ITER.delta
    tau = np.linspace(0, 2 * np.pi, N)
    # Define boundary of ellipse
    x_ellipse = np.asarray(
        [1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau)),
         eps * kappa * np.sin(tau)]
    ).T[::-1]

    if DIVERTOR:
        x_ellipse = v
    X_bc = np.vstack((np.ravel(x_ellipse[:, 0]), np.ravel(x_ellipse[:, 1]))).T

    nlevels = 1000
    cmap = plt.cm.get_cmap("magma", nlevels + 1)

    # Calculate corresponding psi
    # if len(X_test) != 0:
    #     psi_test = []
    #     for point in X_test:
    #         psi_test.append(ITER.psi_func(point[0], point[1]))
    #     psi_true_test = np.reshape(psi_test, [len(psi_test), 1])
    #     output_test = model.predict(X_test)
    #     psi_pred_test = output_test[:, 0].reshape(-1)
    #     psi_pred_test = np.reshape(psi_pred_test, [len(psi_pred_test), 1])
    #     e = (psi_true_test - psi_pred_test) ** 2 / min(psi_true_test) ** 2
    #     print('Average normalized percent error = ', np.mean(np.sqrt(e)) * 100)
    #     print('Max normalized percent error = ', np.max(np.sqrt(e)) * 100)

    # levels = np.logspace(np.log(np.min(error) + 1e-10), np.log(np.max(error)), nlevels + 1)
    # levels = np.linspace(0.0, np.max(error), nlevels + 1)
    cp = ax.contourf(x, y, error + 1e-10, levels=nlevels, norm=LogNorm(), cmap=cmap)
    #cp = ax.contourf(x, y, error, norm=LogNorm(vmin=1e-10), cmap=cmap)
    # cp = ax.contourf(x, y, error, cmap=cmap)

    # Inside
    ax.scatter(
        model.data.train_x[:, 0],
        model.data.train_x[:, 1],
        s=1.5,
        c="#0E0CB5"
    )

    # Boundary
    ax.scatter(
        model.data.bc_points()[:, 0],
        model.data.bc_points()[:, 1],
        s=2,
        c="#D12F24"
    )
    cb = fig.colorbar(cp, ax=ax)  # Add a colorbar to a plot
    # cb.formatter.set_powerlimits((0, 0))
    # cb.ax.yaxis.set_offset_position('right')
    # cb.update_ticks()

    circ = patches.Polygon(xy=X_bc, transform=ax.transData)
    for coll in cp.collections:
        coll.set_clip_path(circ)

    return fig, ax
