def dot(a1, a2):
    return sum(x*y for x, y in zip(a1, a2))

def npdot(v1, v2):
    p = []
    for i in range(len(v2)):
        p.append(dot(v1[i],v2))
    return p

    
def forward_sub(L, b):
    """x = forward_sub(L, b) is the solution to L x = b
       L must be a lower-triangular matrix
       b must be a vector of the same leading dimension as L
    """
    n = len(L)
    x = []

    for t in range(n):
        x.append(0)
    for i in range(n):
        tmp = b[i]
        for j in range(i):
            tmp -= L[i][j] * x[j]
        x[i] = tmp / L[i][i]
    return x


def back_sub(U, b):
    """x = back_sub(U, b) is the solution to U x = b
       U must be an upper-triangular matrix
       b must be a vector of the same leading dimension as U
    """
    n = len(U)
    x = []

    for t in range(n):
        x.append(0)
    for i in range(n-1, -1, -1):
        tmp = b[i]
        for j in range(i+1, n):
            tmp -= U[i][j] * x[j]
        x[i] = tmp / U[i][i]
    return x

def lu_solve(L, U, b):
    """x = lu_solve(L, U, b) is the solution to L U x = b
       L must be a lower-triangular matrix
       U must be an upper-triangular matrix of the same size as L
       b must be a vector of the same leading dimension as L
    """
    y = forward_sub(L, b)
    x = back_sub(U, y)
    return x

# def lup_decomp(A):
#     """(L, U, P) = lup_decomp(A) is the LUP decomposition P A = L U
#        A is any matrix
#        L will be a lower-triangular matrix with 1 on the diagonal, the same shape as A
#        U will be an upper-triangular matrix, the same shape as A
#        U will be a permutation matrix, the same shape as A
#     """
#     n = len(A)
#     if n == 1:
#         L = [[1]]
#         U = A.copy()
#         P = [[1]]
#         return (L, U, P)
#     i = len(A)-1
#     A_bar = np.vstack([A[i,:], A[:i,:], A[(i+1):,:]])

#     A_bar11 = A_bar[0,0]
#     A_bar12 = A_bar[0,1:]
#     A_bar21 = A_bar[1:,0]
#     A_bar22 = A_bar[1:,1:]

#     S22 = A_bar22 - npdot(A_bar21, A_bar12) / A_bar11

#     (L22, U22, P22) = lup_decomp(S22)

#     L11 = 1
#     U11 = A_bar11


#     L12=[]
#     for t in range(n-1):
#         L12.append(0)
#     U12 = A_bar12.copy()

#     L21 = npdot(P22, A_bar21) / A_bar11

#     U21 = []
#     for t in range(n-1):
#         U21.append(0)

#     L = np.block([[L11, L12], [L21, L22]])
#     U = np.block([[U11, U12], [U21, U22]])
#     P = np.block([
#         [np.zeros((1, i-1)), 1,                  np.zeros((1, n-i))],
#         [P22[:,:(i-1)],      np.zeros((n-1, 1)), P22[:,i:]]
#     ])
#     return (L, U, P)

def lup_solve(L, U, P, b):
    """x = lup_solve(L, U, P, b) is the solution to L U x = P b
       L must be a lower-triangular matrix
       U must be an upper-triangular matrix of the same shape as L
       P must be a permutation matrix of the same shape as L
       b must be a vector of the same leading dimension as L
    """
    z = npdot(P,b)
    x = lu_solve(L, U, z)
    return x


