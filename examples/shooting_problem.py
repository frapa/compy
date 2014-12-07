# -*- encoding: utf-8 -*-

#===============================================================================
# INFINITE SQUARE WELL WITH COMPARISON WITH ANALITICAL SOLUTIONS
#-------------------------------------------------------------------------------
# The program calculates the first 4 eigenfunctions of the infinite square well
# and plots them side-by-side with the analitical solutions.
#===============================================================================

from math import *
import compy as cp
import numpy as np
import matplotlib.pyplot as plt

# Domain
D = cp.Domain(-5, 5, 2500)
# Potential
V = 0.5*np.linspace(-5, 5, len(D))**2

# Calculate schrodinger
ev, ef = cp.schrodinger.solve_numerov_shooting(D, V, 1, 1, eigen_num=4, dE=1, precision=1e-12)

f2 = plt.figure()

for E in np.linspace(0.5000, 0.50025, 6):
    a = 2.0*(V - E)
    psi = cp.numerov.numerov_integration(D, a, 1, 2)
    plt.plot(D.as_array(), psi , label=str(E), lw=2)
    
plt.xlabel("x")
plt.xlim((-5, 5))
plt.grid(True, lw=2, ls=":")
plt.legend(loc="upper left")

plt.show()
