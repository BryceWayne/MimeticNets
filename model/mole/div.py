import numpy as np


def div(k, m, dx):
    assert (k >= 2), f"Check order of accuracy.\nk={k}"
    assert (k%2 == 0), f"Check k mod 2.\nk={k}"
    assert (m >= 2*k+1), f"Need m >= {2*k+1}."

    n_rows = m+2
    n_cols = m+1    
    D = np.zeros((n_rows, n_cols))
    neighbors = np.zeros(k)
    neighbors[0] = 0.5 - k//2
    for i in range(1, k):
        neighbors[i] = neighbors[i-1] + 1

    A = np.vander(neighbors).T
    b = np.zeros((k, 1))
    b[k-2] = 1
    coeffs = np.linalg.solve(A, b).T[0]

    j = 0
    for i in range(k//2, n_rows - k//2):
        D[i, j:j+k] = coeffs
        j += 1

    p = k//2 - 1
    q = k + 1
    A = np.zeros((p, q))
    for i in range(p):
        neighbors = np.zeros(q)
        neighbors[1] = 0.5 - i
        for j in range(1, q):
            neighbors[j] = neighbors[j-1] + 1

        V = np.vander(neighbors).T
        b = np.zeros((q, 1))
        b[q-2] = 1 # KEY - selector vector "first deriv"
        coeffs = np.linalg.solve(V,b).T[0]
        A[i, :q] = coeffs

    D[1:p+1, :q] = A

    Pp = np.fliplr(np.eye(p))
    Pq = np.fliplr(np.eye(q))
    A = -Pp@A@Pq
    
    D[n_rows-p-1:n_rows-1, n_cols-q:n_cols] = A
    D[0,:] = 0
    D[-1,:] = 0
    return (1/dx)*D


def div2D(k, m, dx, n, dy):
    Dx = div(k, m, dx)
    Dy = div(k, n, dy)
        
    Im = np.zeros((m + 2, m))
    In = np.zeros((n + 2, n))
        
    Im[1:(m+2)-1, :] = np.eye(m)
    In[1:(n+2)-1, :] = np.eye(n)

    Sx = np.kron(In, Dx)
    Sy = np.kron(Dy, Im)

    D = np.hstack([Sx, Sy])
    return D


if __name__ == '__main__':
    from pprint import pprint
    # pprint(div(2, 5, 1)) # DEBUG
    # pprint(div2D(2, 5, 1, 5, 1)) # DEBUG
