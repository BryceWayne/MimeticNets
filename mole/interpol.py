import numpy as np
import scipy.sparse
import scipy as sp


def interpol(m, c):
    assert (m >= 4), f"m={m}>=4"
    assert (c >= 0 and c <= 1), f"0 <= {c} <= 1"
    
    n_rows = m+1
    n_cols = m+2
    I = np.zeros((n_rows, n_cols))
    I[0, 0], I[-1, -1] = 1, 1
    
    # Average between two continuous cells
    avg = np.array([c, 1-c])
    
    j = 1
    for i in range(1, n_rows - 1):
        I[i, j:j+2] = avg
        j += 1

    return I



def interpol2D(m, n, c1, c2):
    Ix = interpol(m, c1)
    Iy = interpol(n, c2)
        
    Im = np.zeros((m + 2, m))
    In = np.zeros((n + 2, n))
        
    Im[1:(m+2)-1, :] = np.eye(m)
    In[1:(n+2)-1, :] = np.eye(n)

    Sx = np.kron(In.T, Ix)
    Sy = np.kron(Iy, Im.T)

    G = np.vstack([Sx, Sy])
    return G


def interpolD(m, c):
    # Returns a m+2 by m+1 one-dimensional interpolator of 2nd-order
    #
    # Parameters:
    #               m : Number of cells
    #               c : Left interpolation coeff.

    # Assertions:
    assert (m >= 4), f"m={m}>=4"
    assert (c >= 0 and c <= 1), f"0 <= {c} <= 1"

    # Dimensions of I:
    n_rows = m+2
    n_cols = m+1
    
    I = np.zeros((n_rows, n_cols))
    
    I[0, 0] = 1
    I[-1, -1] = 1
    
    # Average between two continuous cells
    avg = np.array([c, 1-c])
    
    j = 0
    for i in range(1, n_cols):
        I[i, j:j+2] = avg
        j += 1

    return I


def interpolD2D(m, n, c1, c2):
    # Returns a two-dimensional interpolator of 2nd-order
    # m : Number of cells along x-axis
    # n : Number of cells along y-axis
    # c1 : Left interpolation coeff.
    # c2 : Bottom interpolation coeff.

    Ix = interpolD(m, c1)
    Iy = interpolD(n, c2)
    
    Im = np.zeros((m + 2, m))
    In = np.zeros((n + 2, n))
    
    Im[1:(m+2)-1, :] = np.eye(m, m)
    In[1:(n+2)-1, :] = np.eye(n, n)
    
    Sx = np.kron(In, Ix)
    Sy = np.kron(Iy, Im)
    
    return np.hstack([Sx, Sy])


if __name__ == '__main__':
    from pprint import pprint
    # pprint(interpol(4, 1)) # DEBUG
    # pprint(interpol2D(4, 4, 1, 1)) # DEBUG
    # pprint(interpolD(4, 1)) # DEBUG
    # pprint(interpolD2D(4, 4, 1, 1)) # DEBUG