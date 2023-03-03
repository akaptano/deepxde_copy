from matplotlib import pyplot as plt
import os
import deepxde as dde
import numpy as np
from utils.inputs import *
import timeit

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

ITER = GS_Linear(eps=0.32, kappa=1.7, delta=0.33)
ITER.get_BCs(A=-0.155)
ITER.solve_coefficients()
colors = ['r', 'b', 'g', 'k', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'lime']
sample_sizes = 5*np.logspace(1, 2, 2, endpoint=True).astype(int)

timings = np.zeros((2, 1))
errors = np.zeros((2, 1, 501 * 501))
for i in range(2):
    for j in range(1):
        ######################
        # Optimization PARAM #
        ######################
        param_val = sample_sizes[i]
        n_domain = param_val
        n_boundary = param_val

        # Check whether the specified path exists or not
        PATH = "".join([DIR, str(param_val)])
        isExist = os.path.exists(PATH)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(PATH)
            print("The new directory is created!")

        # Define geom
        spatial_domain = dde.geometry.Ellipse(eps, kappa, delta) 
        # Generate BC data
        x,u = gen_traindata(n_boundary, eps=eps, kappa=kappa, delta=delta )
        bc135 = dde.PointSetBC(x,u)

        data = dde.data.PDE(
            spatial_domain,
            pde_solovev,
            [bc135],
            num_domain=n_domain,
            num_boundary=0,
            num_test=n_test,
            train_distribution="LHS"
        )

        net = dde.maps.FNN([2] + DEPTH * [BREADTH] + [1], AF, "Glorot normal")
        model = dde.Model(data, net)

        # Compile, train and save model
        model.compile(
            "L-BFGS-B",
            loss_weights=[1,LOSSRATIO]
        )
        ts = timeit.default_timer()
        losshistory, train_state = model.train(
            display_every=10, 
            model_save_path=''.join([PATH,"/model"]),
            #callbacks=[dde.callbacks.DropoutUncertainty()]
        )
        te = timeit.default_timer()
        print("Training took %f s\n" % (te - ts))
        timings[i,j] = te - ts
        dde.saveplot(
            losshistory, 
            train_state, 
            issave=True, 
            isplot=False,
            output_dir=PATH
        )

        loss_train = np.sum(losshistory.loss_train, axis=1)
        loss_train_domain = [item[0] for item in losshistory.loss_train]
        loss_train_boundary = [item[1] for item in losshistory.loss_train]
        loss_test = np.sum(losshistory.loss_test, axis=1)

        if j == 0:
            plt.figure(1)
            plt.semilogy(losshistory.steps, loss_train_domain, color=colors[i], label="{0:.1e} domain train loss".format(param_val))
            plt.semilogy(losshistory.steps, loss_train_boundary / param_val, color=colors[i], linestyle='--' , label="{0:.1e} boundary train loss".format(param_val))  # label=str(LOSSRATIO[i]) + " boundary train loss")

        # Evaluation
        from utils.utils import evaluate,evaluate_eq, relative_error_plot
        x,y,psi_pred,psi_true,error=evaluate(ITER,model)
        errors[i, j, :] = np.sqrt(np.ravel(error))
        print(np.mean(error) * 100)
        print(np.max(error) * 100)

plt.xlabel("# Steps")
plt.legend()
plt.grid(True)
plt.savefig(DIR + '/loss_plot.jpg',dpi = 300)

#########################
# Optimization Analysis #
#########################
mean_errors = np.mean(np.mean(errors, axis=-1), axis=-1)
max_errors = np.mean(np.max(errors, axis=-1), axis=-1)
max_std_errors = np.std(np.max(errors, axis=-1), axis=-1)
std_errors = np.std(np.std(errors, axis=-1), axis=-1)
plt.figure(2)
print(sample_sizes.shape, mean_errors.shape, std_errors.shape, max_errors.shape, max_std_errors.shape)
plt.errorbar(sample_sizes, max_errors, max_std_errors,
             markeredgecolor='k', marker='^', label='Max errors')
plt.errorbar(sample_sizes, mean_errors, std_errors,
             markeredgecolor='k', marker='o', label='Mean errors')
plt.grid(True)
ax = plt.gca()
ax.set_xscale("log")
ax.set_yscale("log")
plt.legend()
plt.savefig(DIR + '/param_analysis.jpg', dpi=300)

###################
# Timing Analysis #
###################
mean_timings = np.mean(timings, axis=-1)
std_timings = np.std(timings, axis=-1)
plt.figure(3)
print(sample_sizes.shape, mean_timings.shape, std_timings.shape)
plt.errorbar(sample_sizes, mean_timings, std_timings,
             markeredgecolor='k', marker='o', label='Mean timings')
plt.grid(True)
ax = plt.gca()
ax.set_xscale("log")
ax.set_yscale("log")
plt.legend()
plt.savefig(DIR + '/timing_analysis.jpg', dpi=300)

# Evaluation
from utils.utils import *
X_test = spatial_domain.random_points(333)
plot_summary_figure(ITER, model, X_test=X_test, losshistory = losshistory, loss_ratio = LOSSRATIO, PATH = DIR)
