import numpy as np
from tqdm import tqdm
from pyl1ksvd.irls import (
    irls_sparse_coding,
    irls_sparse_coding_vectorized,
    l1ksvd_dict_update_step,
)


class L1KSVD:
    """
    l1-K-SVD: Robust Dictionary Learning with Simultaneous Update.

    Minimises the l1-penalised objective:

        min_{D, X}  sum_n  ||y_n - D x_n||_1  +  lam * ||x_n||_1
        subject to  ||d_j||_2 = 1  for all j

    The algorithm alternates between:
      1. **Sparse coding** (fixed D):
         Each x_n is estimated via IRLS (Iteratively Reweighted Least Squares),
         then small-magnitude entries are pruned to zero.
      2. **Dictionary update** (fixed support of X):
         Each atom d_k and its corresponding row x_k are updated *simultaneously*
         by applying Algorithm 1 from the paper (IRLS on the rank-1 residual).

    This formulation assumes a Laplacian prior on both the additive noise and
    the coefficient vector, making the method robust to non-Gaussian/impulsive
    noise (e.g. Laplacian noise, sparse corruption).

    Reference
    ---------
    Mukherjee S., Basu R., Seelamantula C.S.
    "l1-K-SVD: A robust dictionary learning algorithm with simultaneous update."
    Signal Processing 123 (2016) 42–52.

    Parameters
    ----------
    K : int
        Number of dictionary atoms (overcomplete columns).
    lam : float
        l1 regularization parameter lambda.  Larger values promote sparser X.
        Typical range used in the paper: 0.5 – 2.0.
    max_iter : int
        Maximum number of outer dictionary-learning iterations.
    n_irls_sparse : int
        Number of IRLS iterations in the sparse-coding step (per outer iter).
    n_irls_dict : int
        Number of IRLS iterations inside Algorithm 1 (dictionary update, J).
    threshold_frac : float
        After sparse coding, prune entries |x| < threshold_frac * ||X||_F.
        The paper uses 0.03. Set to 0 to disable pruning.
    eps : float
        Small positive constant added for numerical stability (avoids /0).
    """

    def __init__(
        self,
        K=3,
        lam=1.0,
        max_iter=80,
        n_irls_sparse=30,
        n_irls_dict=10,
        threshold_frac=0.03,
        eps=1e-6,
    ):
        self.K = K
        self.lam = lam
        self.MAX_ITER = max_iter
        self.n_irls_sparse = n_irls_sparse
        self.n_irls_dict = n_irls_dict
        self.threshold_frac = threshold_frac
        self.eps = eps

        self.n = None   # signal dimension
        self.N = None   # number of training signals
        self.D = None   # dictionary  (n x K)
        self.X = None   # sparse codes (K x N)
        self.E = None   # residual matrix

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matmul(A, B):
        """Matrix multiply with numerical-stability warnings suppressed."""
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            return A @ B

    @staticmethod
    def _safe_norm(M):
        """Frobenius/vector norm; returns a finite float (falls back to 1e10)."""
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            val = float(np.linalg.norm(M))
        return val if np.isfinite(val) else 1e10

    def _init_dict(self, Y):
        """Initialise dictionary from K randomly chosen (normalised) columns of Y."""
        n, N = Y.shape
        idxs = np.random.choice(N, self.K, replace=(self.K > N))
        D = Y[:, idxs].copy().astype(float)
        norms = np.linalg.norm(D, axis=0)
        norms[norms < self.eps] = 1.0
        D /= norms
        return D

    def _dict_update(self, Y, D, X):
        """
        Update every atom in D (and the corresponding row of X) using
        Algorithm 1 (simultaneous l1-IRLS update).

        Works for any number of atoms K_curr = D.shape[1], so it can be
        called with the full dictionary or with the sub-dictionary in
        fit_with_mean (where the first mean atom is held fixed).

        Parameters
        ----------
        Y : ndarray (n, N)   – training signals (or centred signals)
        D : ndarray (n, K_curr) – current dictionary
        X : ndarray (K_curr, N) – current sparse codes

        Returns
        -------
        D, X  (updated in-place copies)
        """
        n, N = Y.shape
        K_curr = D.shape[1]

        for k in range(K_curr):
            # Error matrix without the contribution of atom k
            E_k = Y - self._matmul(D, X) + np.outer(D[:, k], X[k, :])  # (n, N)

            # Support: signals whose k-th coefficient is non-zero
            indices = np.where(X[k, :] != 0)[0]

            if len(indices) == 0:
                # Atom is unused – reinitialise randomly
                new_atom = np.random.randn(n)
                D[:, k] = new_atom / (np.linalg.norm(new_atom) + self.eps)
                continue

            # Restrict to support columns
            E_k_R = E_k[:, indices]   # (n, |Omega_k|)

            # --- Algorithm 1: simultaneous atom + coefficient update ---
            u, v = l1ksvd_dict_update_step(
                E_k_R,
                lam=self.lam,
                n_iter=self.n_irls_dict,
                eps=self.eps,
            )

            D[:, k] = u
            X[k, indices] = v

        return D, X

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, Y, verbose=False):
        """
        Fit the l1-K-SVD model to training data Y.

        Parameters
        ----------
        Y : ndarray, shape (n, N)
            Training signals (columns).
        verbose : bool
            Print per-iteration error every 5 steps.
        """
        Y = Y.copy().astype(float)
        print("Fitting l1-K-SVD model ...")

        n, N = Y.shape
        self.n = n
        self.N = N

        # Initialisation
        self.D = self._init_dict(Y)
        self.X = np.zeros((self.K, N))
        self.E = Y - self._matmul(self.D, self.X)
        error_t = self._safe_norm(self.E)
        print(f"Initial error: {error_t:.4f}")

        pbar = tqdm(range(self.MAX_ITER), desc=f"Current error: {error_t:.4f}")

        for iteration in pbar:
            # ---- Sparse coding ----
            X_new = irls_sparse_coding_vectorized(
                self.D,
                Y,
                lam=self.lam,
                n_iter=self.n_irls_sparse,
                eps=self.eps,
                threshold_frac=self.threshold_frac,
            )

            X_new = np.nan_to_num(X_new, nan=0.0, posinf=0.0, neginf=0.0)
            err_new = self._safe_norm(Y - self._matmul(self.D, X_new))
            if err_new < error_t:
                self.X = X_new
            else:
                tqdm.write(
                    f"Iter {iteration}: sparse coding did not reduce error "
                    f"({err_new:.4f} >= {error_t:.4f}); keeping old X"
                )

            # ---- Dictionary update (Algorithm 1) ----
            self.D, self.X = self._dict_update(Y, self.D, self.X)

            self.E = Y - self._matmul(self.D, self.X)
            error_t_plus_1 = self._safe_norm(self.E)
            pbar.set_description(f"Current error: {error_t_plus_1:.4f}")

            if verbose and iteration % 5 == 0:
                tqdm.write(f"Error at iteration {iteration}: {error_t_plus_1:.4f}")

            if np.abs(error_t - error_t_plus_1) < 1e-5:
                tqdm.write(
                    f"Converged at iteration {iteration}, "
                    f"delta: {np.abs(error_t - error_t_plus_1):.2e}"
                )
                break

            error_t = error_t_plus_1

        print(f"Final error after fitting: {error_t:.4f}")

    def fit_with_mean(self, Y, verbose=False):
        """
        Fit with a fixed mean atom as the first column of D.

        The first atom is d_0 = (1/sqrt(n)) * 1  (constant vector).
        Its coefficient x_0_n = mean(y_n) * sqrt(n) accounts for the DC
        component; only the remaining K-1 atoms are learned.
        """
        Y = Y.copy().astype(float)
        print("Fitting l1-K-SVD model with mean ...")

        n, N = Y.shape
        self.n = n
        self.N = N

        self.D = np.zeros((n, self.K))
        self.D[:, 0] = 1.0 / np.sqrt(n)          # fixed mean atom

        self.X = np.zeros((self.K, N))
        self.X[0, :] = np.nanmean(Y, axis=0) * np.sqrt(n)

        # Centred signals
        Y_centered = Y - np.outer(self.D[:, 0], self.X[0, :])

        # Initialise the K-1 remaining atoms (replace=True when K-1 > N)
        D_rest = Y_centered[:, np.random.choice(N, self.K - 1, replace=(self.K - 1 > N))].copy()
        norms = np.linalg.norm(D_rest, axis=0)
        norms[norms < self.eps] = 1.0
        D_rest /= norms
        self.D[:, 1:] = D_rest

        X_rest = np.zeros((self.K - 1, N))
        error_t = self._safe_norm(Y_centered - self._matmul(self.D[:, 1:], X_rest))
        pbar = tqdm(range(self.MAX_ITER), desc=f"Current error: {error_t:.4f}")

        for iteration in pbar:
            # ---- Sparse coding on centred signals ----
            X_rest_new = irls_sparse_coding_vectorized(
                self.D[:, 1:],
                Y_centered,
                lam=self.lam,
                n_iter=self.n_irls_sparse,
                eps=self.eps,
                threshold_frac=self.threshold_frac,
            )

            X_rest_new = np.nan_to_num(X_rest_new, nan=0.0, posinf=0.0, neginf=0.0)
            err_new = self._safe_norm(Y_centered - self._matmul(self.D[:, 1:], X_rest_new))
            if err_new < error_t:
                X_rest = X_rest_new

            # ---- Dictionary update on non-mean atoms ----
            self.D[:, 1:], X_rest = self._dict_update(
                Y_centered, self.D[:, 1:], X_rest
            )

            self.E = Y_centered - self._matmul(self.D[:, 1:], X_rest)
            error_t_plus_1 = self._safe_norm(self.E)
            pbar.set_description(f"Current error: {error_t_plus_1:.4f}")

            if verbose and iteration % 5 == 0:
                tqdm.write(f"Error at iteration {iteration}: {error_t_plus_1:.4f}")

            if np.abs(error_t - error_t_plus_1) < 1e-5:
                tqdm.write(f"Converged at iteration {iteration}")
                break

            error_t = error_t_plus_1

        # Store final sparse codes (mean row + learned rows)
        self.X[1:, :] = X_rest
        self.E = Y - self._matmul(self.D, self.X)
        print(f"Final error after fitting: {self._safe_norm(self.E):.4f}")

    def transform(self, Y):
        """
        Compute sparse codes for new signals using the learned dictionary.

        Parameters
        ----------
        Y : ndarray, shape (n, N_t)

        Returns
        -------
        X_t : ndarray, shape (K, N_t)
        D   : ndarray, shape (n, K)   – learned dictionary
        """
        Y = Y.copy().astype(float)
        n_t, N_t = Y.shape
        assert n_t == self.n, (
            f"Expected signals of length {self.n}, got {n_t}"
        )
        print("Transforming data ...")

        X_t = irls_sparse_coding_vectorized(
            self.D,
            Y,
            lam=self.lam,
            n_iter=self.n_irls_sparse,
            eps=self.eps,
            threshold_frac=self.threshold_frac,
        )

        E_t = Y - self._matmul(self.D, X_t)
        print(f"Error after transformation: {self._safe_norm(E_t):.4f}")
        return X_t, self.D

    def transform_with_mean(self, Y):
        """
        Transform with mean subtraction using the fixed first atom.

        Parameters
        ----------
        Y : ndarray, shape (n, N_t)

        Returns
        -------
        X_t : ndarray, shape (K, N_t)
        D   : ndarray, shape (n, K)
        """
        Y = Y.copy().astype(float)
        n_t, N_t = Y.shape
        assert n_t == self.n, (
            f"Expected signals of length {self.n}, got {n_t}"
        )
        print("Transforming data with mean ...")

        X_t = np.zeros((self.K, N_t))
        X_t[0, :] = np.mean(Y, axis=0) * np.sqrt(self.n)

        Y_centered = Y - np.outer(self.D[:, 0], X_t[0, :])

        X_t[1:, :] = irls_sparse_coding_vectorized(
            self.D[:, 1:],
            Y_centered,
            lam=self.lam,
            n_iter=self.n_irls_sparse,
            eps=self.eps,
            threshold_frac=self.threshold_frac,
        )

        E_t = Y - self._matmul(self.D, X_t)
        print(f"Error after transformation: {self._safe_norm(E_t):.4f}")
        return X_t, self.D

    def fit_transform(self, Y):
        """Fit then transform (no mean)."""
        self.fit(Y)
        return self.transform(Y), self.D

    def fit_transform_with_mean(self, Y):
        """Fit then transform (with mean atom)."""
        self.fit_with_mean(Y)
        return self.transform_with_mean(Y), self.D

    def transform_with_mean_signal_with_null_values(self, Y):
        """
        Transform signals that may contain NaN or zero (missing) values.

        For each signal, only the non-null entries are used when computing
        the sparse code.  The mean atom coefficient is estimated from the
        non-null entries.

        Parameters
        ----------
        Y : ndarray, shape (n, N_t)

        Returns
        -------
        X_t : ndarray, shape (K, N_t)
        D   : ndarray, shape (n, K)
        """
        Y = Y.copy().astype(float)
        n_t, N_t = Y.shape

        X_t = np.zeros((self.K, N_t))

        for i in range(N_t):
            Y_i = Y[:, i]
            non_null_idx = np.where(~np.isnan(Y_i) & (Y_i != 0))[0]
            Y_i_non_null = Y_i[non_null_idx]

            if len(non_null_idx) == 0:
                print(f"Warning: signal {i} has no non-null values. Skipping.")
                continue

            mean_val = np.nanmean(Y_i_non_null)
            X_t[0, i] = mean_val * np.sqrt(self.n)

            Y_i_centered = Y_i_non_null - mean_val

            # Sub-dictionary restricted to non-null rows
            D_i = self.D[non_null_idx, 1:].copy()

            x_i = irls_sparse_coding(
                D_i,
                Y_i_centered,
                lam=self.lam,
                n_iter=self.n_irls_sparse,
                eps=self.eps,
            )
            X_t[1:, i] = x_i

        E = Y - self._matmul(self.D, X_t)
        print(f"Error after transformation: {self._safe_norm(E):.4f}")
        return X_t, self.D
