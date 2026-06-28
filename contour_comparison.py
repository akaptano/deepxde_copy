import matplotlib.pyplot as plt
import numpy as np
tau = np.linspace(0, 2 * np.pi, 100)

eps = 0.14328744826120388
kappa = 1.724866877949367
delta = 0.31244999787656796

eps = -1.162e-01
kappa = 1.741e+00
delta = 3.118e-01

eps = 3.135e-01
kappa = 4.349e-01
delta = 2.656e-01

eps = 1.705e-01
kappa = 2.750e+00
delta = 3.465e-01

# eps = 2.654e-01
# kappa = 2.750e+00
# delta = 9.988e-02

eps = 0.229
kappa = 2.647
delta = -0.098

    

x = 1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau))
y = eps * kappa * np.sin(tau)
contour1 = np.column_stack((x, y))

eps = 0.398
kappa = 2.264
delta = 0.444

x = 1 + eps * np.cos(tau + np.arcsin(delta) * np.sin(tau))
y = eps * kappa * np.sin(tau)
contour2 = np.column_stack((x, y))

plt.axis('equal')
plt.plot(contour1[:, 0], contour1[:, 1], label='final contour')
plt.plot(contour2[:, 0], contour2[:, 1], label='initial guess contour')
plt.legend()
plt.show()