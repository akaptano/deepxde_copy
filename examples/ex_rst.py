import numpy as np
import h5py
import matplotlib.pyplot as plt

# Load data from file
with h5py.File("Spheromak-flat_lam-flat_press/psi_gs-500.rst", 'r') as fid:
    # Grid and poloidal flux
    lc_plot = np.asarray(fid['mesh/lc_plot'])
    r_plot = np.asarray(fid['mesh/r_plot'])
    psi = np.asarray(fid['gs/psi'])
    # Profile specifications
    psi_bounds = np.asarray(fid['gs/bounds'])
    p_profile = np.asarray(fid['/gs/p/sample'])*fid['/gs/pnorm']
    f_profile = np.asarray(fid['/gs/f/sample'])*fid['/gs/alam']
    psi_sample = np.linspace(psi_bounds[0], psi_bounds[1], p_profile.shape[0])

# Plot poloidal flux
fig, ax = plt.subplots(1,1)
press_grid = np.interp(psi, psi_sample, p_profile[:,1], left=0.0, right=p_profile[-1,1])
ax.tricontourf(r_plot[:,0], r_plot[:,1], lc_plot, press_grid, 40)
ax.tricontour(r_plot[:,0], r_plot[:,1], lc_plot, psi, colors='k')
ax.set_aspect('equal','box','C')

# Plot F and F' profiles
fig, ax = plt.subplots(2,1)
for i, label in enumerate(("F'", "F")):
    ax[i].plot(psi_sample, f_profile[:,i])
    ax[i].grid(True)
    ax[i].set_ylabel(label)

# Plot P and P' profiles
fig, ax = plt.subplots(2,1)
for i, label in enumerate(("P'", "P")):
    ax[i].plot(psi_sample, p_profile[:,i])
    ax[i].grid(True)
    ax[i].set_ylabel(label)

# Show
plt.show()
