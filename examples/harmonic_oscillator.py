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

# Plot the result
f = plt.figure(figsize=(12, 9))

n = len(ev)
for i, (v, f) in enumerate(zip(ev, ef)):
    ax = plt.subplot(n/2, 2, i+1, axisbelow='True')
    plt.grid(True, lw=2, ls=":")
    
    # plot analitical solution
    #psi = 2/a * np.sin((i+1)*pi/a * D.as_array())**2
    #eigen = ((i+1)*pi/a)**2 / 2
    #plt.plot(D.as_array(), psi, label="{}".format(round(eigen, 4)), lw=7, c="orange")
    
    # plot numerical solution
    sqm = cp.schrodinger.square_modulus(f)
    plt.plot(D.as_array(), sqm, label="{}".format(round(v, 4)), lw=2, c="black")
    
    plt.title(str(i+1), fontsize=16)
    plt.xlabel("x", fontsize=15)
    plt.ylabel(u"|Ψ|²", fontsize=15)
    m = np.max(sqm)
    plt.ylim((-m*0.1, m*1.1))
    plt.legend()

plt.subplots_adjust(left=0.08, bottom=0.07, right=0.95, top=0.95, hspace=0.25)

plt.show()
