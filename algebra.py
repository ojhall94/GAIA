from sympy import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    xlo = 0.795676986895
    xhi = 2.44879973203
    ylo = 4171.36199658
    yhi = 5160.89190244

    labels_mc = ["$m$", "$c$", r"$\sigma(m)$","$lam$"]
    p = np.array([4.60535762e+02, 3.92085724e+03, 6.98355734e+01 , 1.])


    fig, ax = plt.subplots()
    plt.show()
    sys.exit()

    init_printing(use_unicode=False, wrap_line=False, no_global=True)
    x, y, c, m, lam, sig = symbols(['x', 'y', 'c', 'm','lam', 'sig'])

    # fn = exp(lam*x)
    # Ix = integrate(fn, (x, xlo, xhi))
    #
    # Intx = lambdify((lam), Ix)
    # A = 1/Intx(p[-1])

    fn = -0.5*((y - c - m*x)/sig)**2
    Ii = integrate(exp(fn), (x,-oo,oo), (y,-oo,oo))
    # Iy = integrate(exp(fn), (x,xlo,xhi), (y,ylo,yhi))

    print(I)


    # Int = lambdify((c, m, lam, sig), I)
    # C = Int(*p)


    print(integrate(c*x*y, (x,1,2),(y,2,3)))
