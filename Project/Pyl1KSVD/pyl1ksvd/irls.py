import numpy as np


# ---------------------------------------------------------------------------
# Numerical-stability constants
#   w_max   – hard cap on IRLS weights  (prevents overflow in D^T W D products
#             when the residual or coefficient is near zero)
#   _REG    – relative Tikhonov factor added to every linear system
#             (A += _REG * trace(A)/K * I) to keep the system non-singular
# ---------------------------------------------------------------------------
_W_MAX_FACTOR = 1e3   # w_max = _W_MAX_FACTOR / eps  →  1e9 for eps=1e-6;
                      # but we also cap relative to data scale (see _make_wmax)


def _make_wmax(eps):
    """Upper bound for IRLS weights: 1 / sqrt(eps)."""
    return 1.0 / np.sqrt(max(eps, 1e-12))


def irls_sparse_coding(D, y, lam, n_iter=30, eps=1e-6):
    """
    IRLS-based l1 sparse coding for a single signal.

    Solves:  min_x  ||y - D x||_1 + lam * ||x||_1

    The problem is recast as an iteratively reweighted least-squares (IRLS)
    problem using two sets of diagonal weight matrices:
      W1  (n x n): weights on the data-fidelity residual  ||y - Dx||_1
      W2  (K x K): weights on the coefficient penalty     ||x||_1

    At each iteration:
      W1_j = clip(1 / (|r_j| + eps),  0, w_max)    r = y - D x
      W2_j = clip(1 / (|x_j| + eps),  0, w_max)
      x    = (D^T W1 D + lam W2 + reg*I)^{-1}  D^T W1 y

    Parameters
    ----------
    D : ndarray, shape (n, K)
        Dictionary with unit-norm columns.
    y : ndarray, shape (n,)
        Input signal.
    lam : float
        Regularization strength (lambda in the paper).
    n_iter : int
        Number of IRLS iterations.
    eps : float
        Small constant for numerical stability (avoids division by zero).

    Returns
    -------
    x : ndarray, shape (K,)
        Sparse coefficient vector.
    """
    n, K = D.shape
    w_max = _make_wmax(eps)

    # ---- Warm-start -------------------------------------------------------
    # When K >= n the dictionary has full row rank and lstsq gives an exact
    # fit (r ≈ 0), which would set w1 = 1/eps everywhere in the first step
    # and cause BLAS overflow.  We therefore initialise from the minimum-norm
    # lstsq solution but CLIP it so the residual stays non-trivial.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        x, _, _, _ = np.linalg.lstsq(D, y, rcond=None)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # Clamp initial x so |r| is at least ~ ||y||/n (prevents w1 blow-up)
        x_scale = np.linalg.norm(y) / (np.linalg.norm(D @ x) + eps)
        if x_scale < 1.0:          # lstsq over-shoots in overcomplete case
            x *= x_scale

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for _ in range(n_iter):
            r = y - D @ x                              # residual  (n,)

            # Capped IRLS weights
            w1 = np.minimum(1.0 / (np.abs(r) + eps), w_max)   # (n,)
            w2 = np.minimum(1.0 / (np.abs(x) + eps), w_max)   # (K,)

            # Weighted normal equations:  A x = b
            #   A = D^T W1 D + lam W2 + reg I
            #   b = D^T W1 y
            Dw1 = D * w1[:, None]                      # (n, K)
            A = Dw1.T @ D + lam * np.diag(w2)         # (K, K)

            # Tikhonov regularisation – scale with diagonal to stay relative
            reg = eps * (np.trace(A) / K + eps)
            A += reg * np.eye(K)

            b = Dw1.T @ y                              # (K,)

            try:
                x = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

            # Sanitise: kill any NaN / Inf that might have slipped through
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return x


def irls_sparse_coding_vectorized(D, Y, lam, n_iter=30, eps=1e-6, threshold_frac=0.03):
    """
    IRLS-based l1 sparse coding for a matrix of signals.

    Solves, for each column n of Y:
        min_{x_n}  ||y_n - D x_n||_1 + lam * ||x_n||_1

    After convergence, entries with |x| < threshold_frac * ||X||_F are pruned
    to zero (the T0 threshold of the paper; set threshold_frac=0 to disable).

    Parameters
    ----------
    D : ndarray, shape (n, K)
        Dictionary matrix.
    Y : ndarray, shape (n, N)
        Training signals stacked column-wise.
    lam : float
        Regularization parameter.
    n_iter : int
        Number of IRLS iterations per outer loop (applied jointly to all N).
    eps : float
        Numerical stability constant.
    threshold_frac : float
        Pruning threshold fraction.  Set to 0 to disable.

    Returns
    -------
    X : ndarray, shape (K, N)
        Sparse code matrix.
    """
    n, K = D.shape
    N = Y.shape[1]
    w_max = _make_wmax(eps)

    # ---- Warm-start (same stabilisation as the single-signal version) -----
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        X, _, _, _ = np.linalg.lstsq(D, Y, rcond=None)   # (K, N)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        # Scale down if overcomplete to keep initial residuals non-trivial
        DX_norm = np.linalg.norm(D @ X, axis=0) + eps     # (N,)
        Y_norm  = np.linalg.norm(Y, axis=0) + eps          # (N,)
        scale   = np.minimum(Y_norm / DX_norm, 1.0)       # (N,) ≤ 1
        X *= scale[None, :]

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for _ in range(n_iter):
            R = Y - D @ X          # (n, N)

            for i in range(N):
                r = R[:, i]        # (n,)
                x = X[:, i]        # (K,)

                w1 = np.minimum(1.0 / (np.abs(r) + eps), w_max)    # (n,)
                w2 = np.minimum(1.0 / (np.abs(x) + eps), w_max)    # (K,)

                Dw1 = D * w1[:, None]                               # (n, K)
                A = Dw1.T @ D + lam * np.diag(w2)                  # (K, K)
                reg = eps * (np.trace(A) / K + eps)
                A += reg * np.eye(K)

                b = Dw1.T @ Y[:, i]                                 # (K,)

                try:
                    X[:, i] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    X[:, i], _, _, _ = np.linalg.lstsq(A, b, rcond=None)

                X[:, i] = np.nan_to_num(X[:, i], nan=0.0, posinf=0.0, neginf=0.0)

        # Final global sanitise
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Prune small entries to enforce sparsity
    if threshold_frac > 0:
        xf = np.linalg.norm(X, 'fro')
        if np.isfinite(xf) and xf > 0:
            X[np.abs(X) < threshold_frac * xf] = 0.0

    return X


def l1ksvd_dict_update_step(E_R, lam, n_iter=10, eps=1e-6):
    """
    Algorithm 1 from the l1-K-SVD paper: simultaneous atom and coefficient
    update via IRLS.

    Solves:  min_{u, v}  ||E_R - u v^T||_1   s.t.  ||u||_2 = 1

    where E_R is the error matrix restricted to the support Omega_k,
    u is the new dictionary atom (shape M), and v is the vector of
    sparse coefficients for the support signals (shape s).

    Iteration (t -> t+1):
      W1[n, j] = clip(1 / (|(e_n - v_n u)[j]| + eps),  0, w_max)
      u = normalise( (sum_n v_n^2 W1_n)^{-1} sum_n v_n W1_n e_n )
      v_n = (u^T W1_n u + eps)^{-1} u^T W1_n e_n

    Parameters
    ----------
    E_R : ndarray, shape (M, s)
        Restricted error matrix (M = signal dimension, s = support size).
    lam : float
        Regularization parameter (kept for API compatibility; affects
        coefficient penalty magnitude in the sparse-coding stage).
    n_iter : int
        Number of IRLS iterations (J in the paper, typically 10).
    eps : float
        Numerical stability constant.

    Returns
    -------
    u : ndarray, shape (M,)
        Updated dictionary atom (unit-norm).
    v : ndarray, shape (s,)
        Updated sparse coefficients for the support signals.
    """
    M, s = E_R.shape
    w_max = _make_wmax(eps)

    # SVD initialisation (warm-start at the best rank-1 approximation)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        try:
            U, S, Vt = np.linalg.svd(E_R, full_matrices=False)
            u = U[:, 0]
            v = S[0] * Vt[0, :]
        except np.linalg.LinAlgError:
            u = np.random.randn(M)
            u /= np.linalg.norm(u) + eps
            v = np.random.randn(s)

    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    norm_u = np.linalg.norm(u)
    if norm_u > eps:
        u /= norm_u

    # W1[n, :] holds the diagonal of W1_n  (shape: s x M)
    W1 = np.ones((s, M))

    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        for _ in range(n_iter):
            # ------------------------------------------------------------------
            # Step 1 – Update data-fidelity weights W1_n
            # residuals[n, j] = (e_n - v_n * u)[j]
            # ------------------------------------------------------------------
            residuals = E_R.T - v[:, None] * u[None, :]           # (s, M)
            W1 = np.minimum(1.0 / (np.abs(residuals) + eps), w_max)

            # ------------------------------------------------------------------
            # Step 2 – Update atom u
            # A_diag[j] = sum_n v_n^2 * W1[n, j]
            # b_vec[j]  = sum_n v_n * W1[n, j] * E_R[j, n]
            # u_unnorm  = b_vec / A_diag  (diagonal system)
            # ------------------------------------------------------------------
            v_safe = np.clip(v, -1.0 / eps, 1.0 / eps)            # (s,)  no overflow in v^2
            A_diag = np.sum(v_safe[:, None] ** 2 * W1, axis=0)    # (M,)
            b_vec  = np.sum(v_safe[:, None] * W1 * E_R.T, axis=0) # (M,)

            u_new = b_vec / (A_diag + eps)
            u_new = np.nan_to_num(u_new, nan=0.0, posinf=0.0, neginf=0.0)

            norm_u = np.linalg.norm(u_new)
            if norm_u > eps:
                u = u_new / norm_u

            # ------------------------------------------------------------------
            # Step 4 – Update coefficients v_n
            # u^T W1_n u  = sum_j u_j^2 * W1[n, j]
            # u^T W1_n e_n = sum_j u_j * W1[n, j] * E_R[j, n]
            # v_n = (u^T W1_n u + eps)^{-1} u^T W1_n e_n
            # ------------------------------------------------------------------
            uTW1u = np.sum(u ** 2 * W1, axis=1)                   # (s,)
            uTW1e = np.sum(u[None, :] * W1 * E_R.T, axis=1)      # (s,)

            v = uTW1e / (uTW1u + eps)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    return u, v
