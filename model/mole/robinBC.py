import numpy as np
import scipy.sparse
import scipy as sp
from .grad import grad, grad2D

def robinBC(k, m, dx, a, b):
    # Returns a m+2 by m+2 one-dimensional mimetic boundary operator that 
    # imposes a boundary condition of Robin's type
    #
    # Parameters:
    #                k : Order of accuracy
    #                m : Number of cells
    #               dx : Step size
    #                a : Dirichlet Coefficient
    #                b : Neumann Coefficient

    A = np.zeros((m+2, m+2))
    A[0, 0] = a
    A[-1,-1] = a
    
    B = np.zeros((m+2, m+1))
    B[0, 0] = -b
    B[-1, -1] = b
    
    G = grad(k, m, dx)
    
    return A + B@G




def robinBC2D(k, m, dx, n, dy, a, b):
    # Returns a two-dimensional mimetic boundary operator that 
    # imposes a boundary condition of Robin's type
    #
    # Parameters:
    #                k : Order of accuracy
    #                m : Number of cells along x-axis
    #               dx : Step size along x-axis
    #                n : Number of cells along y-axis
    #               dy : Step size along y-axis
    #                a : Dirichlet Coefficient
    #                b : Neumann Coefficient
    # 1-D boundary operator
    Bm = robinBC(k, m, dx, a, b)
    Bn = robinBC(k, n, dy, a, b)
    
    Im = np.eye(m+2)
    In = np.eye(n+2)

    In[0, 0] = 0
    In[-1, -1] = 0
    
    BC1 = np.kron(In, Bm)
    BC2 = np.kron(Bn, Im)
    
    return BC1 + BC2


if __name__ == '__main__':
    from pprint import pprint
    pprint(robinBC(2, 4, 1, 1, 0)) # DEBUG
    pprint(robinBC2D(2, 4, 1, 4, 1, 1, 0)) # DEBUG
