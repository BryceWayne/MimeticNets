import numpy as np
import scipy.sparse
import scipy as sp
from .div import div, div2D
from .grad import grad, grad2D


def lap(k, m, dx):
# Returns a m+2 by m+2 one-dimensional mimetic laplacian operator
#
# Parameters:
#                k : Order of accuracy
#                m : Number of cells
#               dx : Step size

    D = div(k, m, dx)
    G = grad(k, m, dx)
    return D@G



def lap2D(k, m, dx, n, dy):
# Returns a two-dimensional mimetic laplacian operator
#
# Parameters:
#                k : Order of accuracy
#                m : Number of cells along x-axis
#               dx : Step size along x-axis
#                n : Number of cells along y-axis
#               dy : Step size along y-axis

    D = div2D(k, m, dx, n, dy)
    G = grad2D(k, m, dx, n, dy)
    return D@G


if __name__ == '__main__':
    from pprint import pprint
    # pprint(lap(2, 4, 1)) # DEBUG
    # pprint(lap2D(2, 4, 1, 4, 1)) # DEBUG
