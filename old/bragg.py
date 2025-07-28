import numpy as np
from numpy.linalg import pinv, norm

# helpers
def sample_simplex(k, n, alpha=1.0):
    """n samples from Dirichlet(alpha) on the (k-1)-simplex."""
    return np.random.dirichlet(alpha * np.ones(k), size=n).T  # shape (k, n)

def project_to_simplex(v):
    """
    Euclidean projection of v onto the probability simplex.
    Algorithm: Wang & Carreira-Perpiñán, "Projection onto the simplex" (2013).
    """
    k = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.nonzero(u + cssv / np.arange(1, k+1) > 0)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)

def columnwise_simplex_projection(P):
    """Project each column of P onto the simplex (in-place)."""
    for j in range(P.shape[1]):
        P[:, j] = project_to_simplex(P[:, j])
    return P

def procrustes_align(A, B):
    """
    Orthogonal Procrustes: find R that minimises ||A - B R||_F.
    Returns A_aligned = B R and R.
    """
    U, _, Vt = np.linalg.svd(B.T @ A, full_matrices=False)
    R = U @ Vt
    return B @ R, R

# ---------- synthetic data --------------------------------------------------
rng = np.random.default_rng(seed=42)

d, k, n = 10, 3, 500          # ambient dim, latent dim, #observations
noise_std = 0.02

B0_true = rng.normal(size=(d, k))
P_true  = sample_simplex(k, n)          # shape (k, n)
B       = B0_true @ P_true # + noise_std * rng.normal(size=(d, n))

# ---------- alternating updates ---------------------------------------------
max_iter = 200
tol      = 1e-6

# init P (Dirichlet) and first B0
P = sample_simplex(k, n)
for it in range(max_iter):
    B0 = B @ P.T @ pinv(P @ P.T)        # least-squares update
    
    # store for convergence test
    P_prev = P.copy()
    
    # unconstrained LS then simplex projection, column-wise
    P = pinv(B0) @ B                    # unconstrained
    P = columnwise_simplex_projection(P)
    
    delta = norm(P - P_prev) / (norm(P_prev) + 1e-15)
    if delta < tol:
        print(f'Converged in {it+1} iterations (Δ={delta:.2e})')
        break
else:
    print('Reached max_iter without full convergence.')

# ---------- evaluation ------------------------------------------------------
# align learned B0 with ground truth (up to rotation/permutation)
B0_aligned, R = procrustes_align(B0_true, B0)

basis_error = norm(B0_true - B0_aligned) / norm(B0_true)
P_error     = norm(P_true - R.T @ P) / norm(P_true)

print(f'Basis relative error: {basis_error:.3%}')
print(f'Loadings relative error: {P_error:.3%}')
