function [x, residHist, errHist] = reweighted_l2(A, b, errFcn, opts)
%REWEIGHTED_L2  Chartrand-Yin Iteratively Reweighted L2 for sparse recovery.
%
%   Minimizes the smoothed lp surrogate:
%       min  sum_i ( |x_i|^2 + eps )^(p/2)   subject to  Ax = b
%   by iteratively solving a re-weighted least-squares problem.
%
%   At each outer iteration, weights are fixed at
%       w_i = ( |x_i|^2 + eps )^(p/2 - 1)
%   and the following (analytically solvable) weighted LS problem is solved:
%       min  x' W x   s.t.  Ax = b
%   whose closed-form solution is:
%       x = W^{-1} A' ( A W^{-1} A' )^{-1} b
%   where W = diag(w).  After each solve, eps is decreased geometrically
%   so the surrogate tightens toward the true lp norm.
%
%   Reference:
%     R. Chartrand and W. Yin, "Iteratively reweighted algorithms for
%     compressive sensing," Proc. IEEE ICASSP, Las Vegas, 2008.
%
%   Usage:
%     x = reweighted_l2(A, b)
%     [x, residHist, errHist] = reweighted_l2(A, b, errFcn, opts)
%
%   Inputs:
%     A         - M x N measurement matrix (real or complex)
%     b         - M x 1 observation vector
%     errFcn    - (optional) function handle returning a scalar error,
%                 e.g.  @(xhat) norm(xhat - x_true)/norm(x_true)
%                 Pass [] to skip error tracking.
%     opts      - (optional) struct with any of the following fields:
%
%       .p          Exponent for the lp surrogate, p in (0, 2).
%                   Smaller p => stronger sparsity promotion but harder
%                   optimisation.  p = 0.5 is a good default.
%                   Default: 0.5
%
%       .eps0       Initial value of the smoothing parameter eps.
%                   Should be on the order of the squared signal amplitude.
%                   Default: automatically set to  mean(|x^0|^2) + 1
%
%       .eps_factor Multiplicative factor by which eps is decreased each
%                   outer iteration (0 < eps_factor < 1).
%                   Smaller => faster tightening, but may cause
%                   ill-conditioning if too small.
%                   Default: 0.1
%
%       .eps_min    Floor for eps (stops decreasing below this).
%                   Default: 1e-8
%
%       .maxiter    Maximum number of outer (re-weighting) iterations.
%                   Default: 30
%
%       .tol        Convergence tolerance on the relative change in x:
%                     ||x_new - x_old|| / ||x_old|| < tol  => stop.
%                   Default: 1e-6
%
%       .printEvery Print progress every this many iterations.
%                   Set to Inf to suppress all output.
%                   Default: 5
%
%   Outputs:
%     x         - N x 1 sparse estimate
%     residHist - vector of ||Ax - b|| at each iteration
%     errHist   - vector of errFcn(x)  at each iteration ([] if errFcn=[])
%

% -------------------------------------------------------------------------
%   Parse inputs
% -------------------------------------------------------------------------
if nargin < 3, errFcn = []; end
if nargin < 4, opts   = []; end

if ~isempty(opts) && ~isstruct(opts)
    error('"opts" must be a struct or [].');
end

p          = getopt(opts, 'p',          0.5 );
eps_factor = getopt(opts, 'eps_factor', 0.1 );
eps_min    = getopt(opts, 'eps_min',    1e-8);
maxiter    = getopt(opts, 'maxiter',    30  );
tol        = getopt(opts, 'tol',        1e-6);
printEvery = getopt(opts, 'printEvery', 5   );

if ~isempty(errFcn) && ~isa(errFcn, 'function_handle')
    error('"errFcn" must be a function handle or [].');
end
if p <= 0 || p >= 2
    error('"p" must satisfy 0 < p < 2.');
end

[M, N] = size(A);

% -------------------------------------------------------------------------
%   Initialise: x^0 = minimum l2-norm solution  A^+ b
% -------------------------------------------------------------------------
% Using the normal equations for the underdetermined case (M < N):
%   x^0 = A' (A A')^{-1} b
AAt   = A * A';                     % M x M
x     = A' * (AAt \ b);            % N x 1   (real or complex)

% Set eps0 automatically if not supplied by user
if isfield(opts, 'eps0') && ~isempty(opts.eps0)
    eps_k = opts.eps0;
else
    eps_k = mean(abs(x).^2) + 1;   % scales with initial energy
end

residHist = zeros(maxiter, 1);
errHist   = zeros(maxiter, 1);

% -------------------------------------------------------------------------
%   Print header
% -------------------------------------------------------------------------
if printEvery < Inf
    if ~isempty(errFcn)
        fprintf('Iter,   eps,        Resid,      Error\n');
    else
        fprintf('Iter,   eps,        Resid\n');
    end
end

% -------------------------------------------------------------------------
%   Main reweighting loop
% -------------------------------------------------------------------------
for k = 1:maxiter

    x_old = x;

    % --- Step 1: Compute diagonal weights ---
    %   w_i = ( |x_i|^2 + eps )^(p/2 - 1)
    %   Since p < 2, the exponent (p/2 - 1) < 0, so larger |x_i| => smaller w_i
    %   => the weighted LS objective penalises small |x_i| more, driving them to 0.
    w    = (abs(x).^2 + eps_k).^(p/2 - 1);   % N x 1,  w_i > 0

    % --- Step 2: Solve weighted LS with equality constraint ---
    %   min  x' W x   s.t.  Ax = b
    %   Analytic solution:  x = W^{-1} A' ( A W^{-1} A' )^{-1} b
    %
    %   Numerically: let s = 1./w  (diagonal of W^{-1}), then
    %     A W^{-1} A'  =  A * diag(s) * A'  =  (s' .* A) * A'   (M x M)
    %   This avoids forming the full N x N matrix W^{-1}.

    s       = 1 ./ w;                           % N x 1 scaling vector
    % A W^{-1} A' = A * diag(s) * A'
    %   s .* A'  scales each row i of A' (N x M) by s(i)  => N x M
    %   A * (s .* A')  is  M x N times N x M  =>  M x M
    AWinvAt = A * (s .* A');                    % M x M

    % Solve the M x M system and form x:
    %   x = diag(s) * A' * (AWinvAt \ b)
    x = s .* (A' * (AWinvAt \ b));             % N x 1

    % --- Step 3: Decrease eps (tighten the lp approximation) ---
    eps_k = max(eps_k * eps_factor, eps_min);

    % --- Step 4: Diagnostics ---
    resid = norm(A * x - b);
    residHist(k) = resid;

    PRINT = (printEvery < Inf) && (~mod(k, printEvery) || k == maxiter);

    if ~isempty(errFcn)
        er = errFcn(x);
        errHist(k) = er;
        if PRINT
            fprintf('%4d,  %.3e,  %.3e,  %.3e\n', k, eps_k, resid, er);
        end
    else
        if PRINT
            fprintf('%4d,  %.3e,  %.3e\n', k, eps_k, resid);
        end
    end

    % --- Step 5: Check convergence ---
    rel_change = norm(x - x_old) / (norm(x_old) + 1e-300);
    if rel_change < tol
        if printEvery < Inf
            fprintf('Converged at iter %d  (rel. change in x = %.2e < tol = %.2e)\n', ...
                    k, rel_change, tol);
        end
        residHist = residHist(1:k);
        errHist   = errHist(1:k);
        break;
    end

end % for k

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
