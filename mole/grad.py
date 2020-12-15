import numpy as np
import scipy.sparse
import scipy as sp


def grad(k, m, dx):
    # Returns a m+1 by m+2 one-dimensional mimetic gradient operator
    #
    # Parameters:
    #                k : Order of accuracy
    #                m : Number of cells
    #               dx : Step size
    assert (k >= 2), f"Check order of accuracy.\nk={k}"
    assert (k%2 == 0), f"Check k mod 2.\nk={k}"
    assert (m >= 2*k), f"m must be greater, or equal to, than 2*k.\n{m} must be greater, or equal to, than {2*k}."
    
    n_rows = m+1
    n_cols = m+2
    G = np.zeros((n_rows, n_cols))
    neighbors = np.zeros(k)
    neighbors[0] = 0.5 - k/2
    for i in range(1, k):
        neighbors[i] = neighbors[i-1] + 1

    A = np.vander(neighbors).T
    b = np.zeros((k, 1))
    b[k-2] = 1    
    coeffs = np.linalg.solve(A,b).T[0]
    
    j = 1
    for i in range(k//2, n_rows - k//2):
        G[i, j:j+k] = coeffs
        j += 1

    p = k//2
    q = k + 1
    A = np.zeros((p, q))
    for i in range(p):
        neighbors = np.zeros(q)
        neighbors[0] = i
        neighbors[1] = neighbors[0] + 0.5
        for j in range(2, q):
            neighbors[j] = neighbors[j-1] + 1
        
        V = np.vander(neighbors).T
        b = np.zeros((q, 1))
        b[q-2] = 1
        coeffs = np.linalg.solve(V, b).T[0]
        A[i, :q] = coeffs
        
    G[:p, :q] = A
    
    Pp = np.fliplr(np.eye(p))
    Pq = np.fliplr(np.eye(q))
    
    A = -Pp@A@Pq
    
    G[n_rows-p:n_rows, n_cols-q:n_cols] = A
    return (1/dx)*G



def grad2D(k, m, dx, n, dy):
    Gx = grad(k, m, dx)
    Gy = grad(k, n, dy)
        
    Im = np.zeros((m + 2, m))
    In = np.zeros((n + 2, n))
        
    Im[1:(m+2)-1, :] = np.eye(m)
    In[1:(n+2)-1, :] = np.eye(n)

    Sx = np.kron(In.T, Gx)
    Sy = np.kron(Gy, Im.T)

    G = np.vstack([Sx, Sy])
    return G


if __name__ == '__main__':
    from pprint import pprint
    # pprint(np.round(grad(2, 4, 1), 2)) # DEBUG
    G = grad2D(2, 4, 1, 4, 1)
    print(G.shape)
    pprint(G) # DEBUG

