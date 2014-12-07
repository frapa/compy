import numpy

from . import numerov
from . import integrals

def square_modulus(psi):
    return psi * numpy.conjugate(psi)

def solve_numerov_matching(D, V, boundary_start, boundary_end, m=1.0, h=1.0, dE=0.1, E_min=None, E_max=10.0,
        eigen_num=None, precision=0.0001, normalized=True, callback=None):
    
    # Find minimum of potential
    if E_min is None:
        E_min = numpy.min(V)
    else:
        Em = numpy.min(V)
        if E_min < Em:
            E_min = Em

    # Find eigenvalues of energy using the shooting method
    # The following lines allow the function to accept both a range of energies
    # or a number of eigenvalues to be found
    if eigen_num is None:
        energies = numpy.linspace(E_min, E_max, int((E_max - E_min)/dE))
    else:
        # Generator of energies. In the following for loop
        # we will exit when we have found eigen_num number of eigenvalues
        def energies():
            n = 0
            while True:
                yield E_min + dE * n
                n += 1

        energies = energies()

    eigenvalues = []
    eigenfunctions = []
    # Error on wave function. It is not significant since most times we are 
    # not integrating with known boundary conditions.
    wf_err = None 
    last_E = None
    for n, E in enumerate(energies):
        a = 2.0*m*(V - E) / h**2

        psi = numerov.numerov_integration(D, a, boundary_start, boundary_start+1.0)
        new_wf_err = psi[-1] - boundary_end

        if wf_err is not None and wf_err*new_wf_err < 0:
            pass

def solve_numerov_shooting(D, V, boundary_start, boundary_end, m=1.0, h=1.0, dE=0.1, E_min=None, E_max=10.0,
        eigen_num=None, precision=0.0001, normalized=True, callback=None):

    # Find minimum of potential
    if E_min is None:
        E_min = numpy.min(V)
    else:
        Em = numpy.min(V)
        if E_min < Em:
            E_min = Em

    # Find eigenvalues of energy using the shooting method
    # The following lines allow the function to accept both a range of energies
    # or a number of eigenvalues to be found
    if eigen_num is None:
        energies = numpy.linspace(E_min, E_max, int((E_max - E_min)/dE))
    else:
        # Generator of energies. In the following for loop
        # we will exit when we have found eigen_num number of eigenvalues
        def energies():
            n = 0
            while True:
                yield E_min + dE * n
                n += 1

        energies = energies()

    eigenvalues = []
    eigenfunctions = []
    # Error on wave function. It is not significant since most times we are 
    # not integrating with known boundary conditions.
    wf_err = None 
    last_E = None
    for n, E in enumerate(energies):
        a = 2.0*m*(V - E) / h**2

        psi = numerov.numerov_integration(D, a, boundary_start, boundary_start+1.0)
        new_wf_err = psi[-1] - boundary_end

        if wf_err is not None and wf_err*new_wf_err < 0:
            # We found a energy eigenvalue!
            # Now, using the shooting method (bisection), we get the wave function
            # according to precision
            
            # E_err instead of wf_err. This time the error is on the energy eigenvalue.
            # Moreover, I use E_ instead of E to avoid modification of E in the outer loop
            E_ = E
            E_err = E_ - last_E
            
            # Moreover we use another variable wf_error instead of wf_err to avoid confusion
            # with the outer loop.
            wf_error = new_wf_err
            
            E1 = last_E
            E2 = E_
            while abs(E_err) > precision:
                E_ = (E2 + E1) / 2.0

                a = 2.0*m*(V - E_) / h**2

                psi = numerov.numerov_integration(D, a, boundary_start, boundary_start+1.0)
                new_wf_error = psi[-1] - boundary_end

                if wf_error*new_wf_error < 0:
                    E1 = E2
                    E2 = E_
                else:
                    E2 = E_

                wf_error = new_wf_error
                E_err = E2 - E1
            
            print(wf_error, new_wf_error, E_)

            eigenvalues.append(E_)
            eigenfunctions.append(psi)
        
        if callback is not None:
            if isinstance(energies, numpy.ndarray):
                fraction = float(n) / len(energies)
            else:
                fraction = float(len(eigenvalues)) / eigen_num
            
            callback(fraction)
        
        if eigen_num is not None and len(eigenvalues) == eigen_num:
            break

        wf_err = new_wf_err
        last_E = E

    # If requested, we normalize the wave functions
    if normalized:
        for n, psi in enumerate(eigenfunctions):
            integral = integrals.trapezioidal(D, psi * numpy.conjugate(psi))
            eigenfunctions[n] = psi / numpy.sqrt(integral)
    
    if callback is not None:
        callback(1.0)

    return (eigenvalues, eigenfunctions)
