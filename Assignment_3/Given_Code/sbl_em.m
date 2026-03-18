function [x, residHist, errHist] = sbl_em(A, b, errFcn, opts)
%SBL_EM  Sparse Bayesian Learning with EM updates for sparse recovery.
%
%   Probabilistic model:
%       b  =  A*x  +  noise,   noise ~ CN(0, sigma2 * I_M)
%       x_i ~ CN(0, gamma_i)   independently  (i = 1,...,N)
%
%   The hyperparameters {gamma_i} (prior variances) are estimated by
%   maximizing the marginal likelihood p(b | gamma, sigma2) via EM.
%   Components with gamma_i -> 0 are automatically pruned, yielding
%   a sparse posterior mean as the estimate of x.
%
%   EM update equations (per iteration):
%   -----------------------------------------------------------------------
%   Let Gamma = diag(gamma),  C = sigma2*I + A*Gamma*A'  (M x M)
%
%   E-step  (posterior statistics of x given b, current gamma, sigma2):
%       mu      = Gamma * A' * C^{-1} * b              (N x 1 posterior mean)
%       Sigma_ii = gamma_i - gamma_i^2 * [A'C^{-1}A]_ii  (posterior var, diag)
%
%   M-step  (update hyperparameters to maximise expected log-likelihood):
%       gamma_i_new  = |mu_i|^2 + Sigma_ii             (Type-II ML update)
%       sigma2_new   = ( ||b - A*mu||^2
%                        + sigma2 * sum_i(1 - gamma_i*[A'C^{-1}A]_ii) ) / M
%
%   Efficient implementation using Cholesky of C (M x M, not N x N):
%       L          = chol(C, 'lower')
%       mu         = gamma .* ( A' * (L' \ (L \ b)) )
%       [A'C^{-1}A]_ii = sum(abs(L \ A).^2, 1)'      (via column norms of L\A)
%   -----------------------------------------------------------------------
%
%   Reference:
%     M. E. Tipping, "Sparse Bayesian Learning and the Relevance Vector
%       Machine," JMLR, vol. 1, pp. 211-244, 2001.
%     D. P. Wipf and B. D. Rao, "Sparse Bayesian Learning for Basis
%       Selection," IEEE Trans. Signal Processing, 52(8), 2004.
%
%   Usage:
%     x = sbl_em(A, b)
%     [x, residHist, errHist] = sbl_em(A, b, errFcn, opts)
%
%   Inputs:
%     A       - M x N measurement matrix (real or complex)
%     b       - M x 1 observation vector
%     errFcn  - (optional) function handle returning a scalar error metric,
%               e.g.  @(xhat) norm(xhat - x_true)/norm(x_true).
%               Pass [] to skip.
%     opts    - (optional) struct with any of the following fields:
%
%       .sigma2        Initial noise variance.
%                      Default: 0.1 * norm(b)^2 / M  (10% of signal energy)
%
%       .update_sigma2 Whether to update sigma2 at each EM iteration.
%                      Set false if you know sigma2 (e.g. noiseless case).
%                      Default: true
%
%       .maxiter       Maximum number of EM iterations.
%                      Default: 500
%
%       .tol           Convergence tolerance: stop when
%                        ||gamma_new - gamma_old|| / ||gamma_old|| < tol.
%                      Default: 1e-6
%
%       .prune_tol     Components with gamma_i < prune_tol are set to zero
%                      and excluded from computation (speeds up later iters).
%                      Default: 1e-10
%
%       .printEvery    Print a status line every this many iterations.
%                      Set to Inf to suppress all output.
%                      Default: 50
%
%   Outputs:
%     x         - N x 1 sparse estimate (posterior mean)
%     residHist - vector of ||Ax - b|| at each iteration
%     errHist   - vector of errFcn(x)  at each iteration ([] if errFcn=[])
%
%   See also OMP, CoSaMP, reweighted_l2

% -------------------------------------------------------------------------
%   Parse inputs and options
% -------------------------------------------------------------------------
if nargin < 3, errFcn = []; end
if nargin < 4, opts   = []; end

if ~isempty(opts) && ~isstruct(opts)
    error('"opts" must be a struct or [].');
end
if ~isempty(errFcn) && ~isa(errFcn, 'function_handle')
    error('"errFcn" must be a function handle or [].');
end

sigma2_init    = getopt(opts, 'sigma2',        []);
update_sigma2  = getopt(opts, 'update_sigma2', true);
maxiter        = getopt(opts, 'maxiter',        500);
tol            = getopt(opts, 'tol',            1e-6);
prune_tol      = getopt(opts, 'prune_tol',      1e-10);
printEvery     = getopt(opts, 'printEvery',     50);

[M, N] = size(A);

% -------------------------------------------------------------------------
%   Column normalisation  (critical when A = phi * FFT_matrix)
%
%   Each column of A = phi*F has norm ~ sqrt(M*N) ~ 200 for typical sizes.
%   Without normalisation, A*diag(gamma)*A' has diagonal entries ~ M*N,
%   making C >> sigma2*I so the posterior mean ~ 0 on iteration 1, sigma2
%   blows up on the first update, and all gamma collapse to zero.
%
%   We normalise each column to unit norm internally and undo the scaling
%   at the end:  x_true = x_normalised ./ col_norms
% -------------------------------------------------------------------------
col_norms = sqrt(sum(abs(A).^2, 1))';   % N x 1, norm of each column
col_norms = max(col_norms, 1e-10);      % guard against zero columns
A         = A .* (1 ./ col_norms)';              % A_norm  (M x N, unit cols)

% -------------------------------------------------------------------------
%   Initialise hyperparameters
% -------------------------------------------------------------------------
% All prior variances equal — no preference for any component initially.
gamma  = ones(N, 1);

% Initial noise variance.
% After column normalisation, A_norm*A_norm' has diagonal ~ N (not M*N),
% so a small fraction of the per-sample observation energy is appropriate.
% For near-noiseless problems this drives quickly to near-zero via the
% sigma2 update; for noisy problems the EM refines it from this starting
% point.
if isempty(sigma2_init)
    sigma2 = 1e-10 * real(b' * b) / M;
else
    sigma2 = sigma2_init;
end
sigma2 = max(sigma2, 1e-12);

residHist = zeros(maxiter, 1);
errHist   = zeros(maxiter, 1);

% -------------------------------------------------------------------------
%   Print header
% -------------------------------------------------------------------------
if printEvery < Inf
    if ~isempty(errFcn)
        fprintf('Iter,  N_active,  sigma2,      Resid,      Error\n');
    else
        fprintf('Iter,  N_active,  sigma2,      Resid\n');
    end
end

% -------------------------------------------------------------------------
%   Main EM loop
% -------------------------------------------------------------------------
for k = 1:maxiter

    gamma_old = gamma;

    % ---- Identify active (non-pruned) components ----
    active = gamma > prune_tol;          % N x 1 logical
    Na     = sum(active);

    if Na == 0
        warning('SBL_EM: all components pruned; returning zero vector.');
        residHist = residHist(1:k-1);
        errHist   = errHist(1:k-1);
        break;
    end

    A_a = A(:, active);                  % M x Na  (restrict columns)
    g_a = gamma(active);                 % Na x 1

    % ================================================================
    %   E-STEP: Compute posterior mean mu and diagonal of posterior cov
    % ================================================================

    % Form C = sigma2*I + A_a * diag(g_a) * A_a'  (M x M)
    %   g_a .* A_a'  scales row i of A_a' by g_a(i)  =>  Na x M
    %   A_a * (g_a .* A_a')  =>  M x M
    C = sigma2 * eye(M) + A_a * (g_a .* A_a');   % M x M, Hermitian PD

    % When A is complex (e.g. A = phi*F with F = FFT matrix), floating-point
    % arithmetic leaves tiny imaginary parts on the diagonal of C even though
    % mathematically C_ii = sigma2 + sum_j gamma_j |A_ij|^2 is real.
    % Enforcing Hermitian symmetry removes those spurious imaginary parts and
    % is required for MATLAB's chol to accept the matrix.
    C = (C + C') / 2;

    % Cholesky factorisation: C = L * L'  (L lower-triangular)
    [L, flag] = chol(C, 'lower');
    if flag ~= 0
        % Still not PD after symmetrisation: add a small ridge and retry
        C  = C + 1e-10 * trace(C) / M * eye(M);
        C  = (C + C') / 2;
        L  = chol(C, 'lower');
    end

    % C^{-1} * b  via two triangular solves
    Cinv_b = L' \ (L \ b);              % M x 1

    % Posterior mean on active set:
    %   mu_a = diag(g_a) * A_a' * C^{-1} * b
    mu_a = g_a .* (A_a' * Cinv_b);     % Na x 1

    % Diagonal of [A' C^{-1} A] for active columns:
    %   [A_a' C^{-1} A_a]_ii = || (L^{-1} A_a)_i ||^2
    %   V = L \ A_a  =>  M x Na
    %   d_a = sum(|V|^2, 1)'  =>  Na x 1
    V   = L \ A_a;                      % M x Na
    d_a = sum(abs(V).^2, 1)';           % Na x 1

    % Diagonal of posterior covariance on active set:
    %   Sigma_ii = gamma_i - gamma_i^2 * d_i
    %            = gamma_i * (1 - gamma_i * d_i)
    Sigma_diag_a = g_a .* (1 - g_a .* d_a);   % Na x 1
    Sigma_diag_a = max(Sigma_diag_a, 0);        % guard against tiny negatives

    % ================================================================
    %   M-STEP: Update gamma and (optionally) sigma2
    % ================================================================

    % Gamma update (Type-II ML / evidence maximisation):
    %   gamma_i_new = E[|x_i|^2 | b] = |mu_i|^2 + Sigma_ii
    gamma_new_a      = abs(mu_a).^2 + Sigma_diag_a;   % Na x 1
    gamma(:)         = 0;
    gamma(active)    = gamma_new_a;

    % Noise variance update — simple ML estimate:
    %   sigma2_new = ||b - A*mu||^2 / M
    %
    % The full Tipping formula adds a trace-correction term that can make
    % sigma2 increase when it should decrease (especially in early iterations
    % with all-ones gamma and large A).  The plain residual-power formula is
    % equivalent to the Wipf-Rao (2004) update and is numerically stable.
    if update_sigma2
        resid_vec = b - A_a * mu_a;
        sigma2    = max(real(resid_vec' * resid_vec) / M, 1e-12);
    end

    % ================================================================
    %   Diagnostics
    % ================================================================
    x_full          = zeros(N, 1);
    x_full(active)  = mu_a;

    resid            = norm(b - A * x_full);
    residHist(k)     = resid;

    PRINT = (printEvery < Inf) && (~mod(k, printEvery) || k == maxiter);
    if ~isempty(errFcn)
        er          = errFcn(x_full);
        errHist(k)  = er;
        if PRINT
            fprintf('%4d,  %6d,    %.3e,  %.3e,  %.3e\n', ...
                    k, Na, sigma2, resid, er);
        end
    else
        if PRINT
            fprintf('%4d,  %6d,    %.3e,  %.3e\n', k, Na, sigma2, resid);
        end
    end

    % ================================================================
    %   Convergence check
    % ================================================================
    rel_change = norm(gamma - gamma_old) / (norm(gamma_old) + 1e-300);
    if rel_change < tol
        if printEvery < Inf
            fprintf('Converged at iter %d  (rel. change in gamma = %.2e)\n', ...
                    k, rel_change);
        end
        residHist = residHist(1:k);
        errHist   = errHist(1:k);
        break;
    end

end % EM loop

% -------------------------------------------------------------------------
%   Final estimate: re-run E-step with converged gamma for clean output
% -------------------------------------------------------------------------
active = gamma > prune_tol;
x      = zeros(N, 1);

if any(active)
    A_a    = A(:, active);
    g_a    = gamma(active);
    C      = sigma2 * eye(M) + A_a * (g_a .* A_a');
    C      = (C + C') / 2;          % enforce Hermitian symmetry (complex A fix)
    x(active) = g_a .* (A_a' * (C \ b));
end

% Undo column normalisation: x_norm solves the problem for A_normalised,
% so x_true = x_norm ./ col_norms
x = x ./ col_norms;

end % main function

% -------------------------------------------------------------------------
%   Helper: read option field, return default if absent
% -------------------------------------------------------------------------
function val = getopt(opts, field, default)
    if isempty(opts) || ~isfield(opts, field)
        val = default;
    else
        val = opts.(field);
    end
end
