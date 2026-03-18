%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% A routine to compare l1, SBL and OMP for sinusoidal signals recovery
% sampled using random linear projections
%
% Implements the example in: S. Engelberg, "Compressive Sensing", IEEE
% Instrumentation and Measurement Magazine, Feb. 2012. 
%
% Dependencies: l1 magic package, OMP package, reweighted_l2.m
%
% Author: Chandra Murthy, ECE Dept., IISc
% Email : cmurthy@iisc.ac.in
% Date last updated : 24 Jan. 2020
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Begin
clear all; close all;

% add relevant paths
% Replace this with the path to the location of CVX
%addpath('../../cvx')
addpath('./l1magic', './l1magic/optimization')

%% Set parameters 

rng(42);
% number of measurements
m = 200;
% number of samples in 1 second, also, length of sparse signal
n = 512;

% List of tones that go into the sinusoidal signal

k = 5; % number of tones; it is a measure of sparsity
f = 256*rand([1,k]); % Hz   
theta = 2*pi*rand([1,k]); % radians
amp = abs(10*randn([1,k])); % value
%amp = 10*ones([1,k]);

% sparsity: length(f) is only the approximate value of sparsity since we
% are going to use windowing
k = length(f);

% time domain samples
y = sum((ones(n,1)*amp).*sin(2*pi*[0:1:n-1]'*f/n + ones(n,1)*theta),2);

% FFT and inverse FFT matrices
F = fft(eye(n));
iF = F'/n;

% apply a Hanning window to smooth out the edges
W = diag(hanning(n));
ywin = W*y;

plot([0:n-1]/n, y, 'r-', [0:n-1]/n, ywin, 'b-');
xlabel('Time');
ylabel('Time-domain Signal');
legend('Original signal', 'windowed signal');

%pause;
%close;

%% Frequency domain
% frequency domain vector, will be more or less sparse
Y = F'*ywin/n;

plot(([0:1:n-1]-n/2)/2, abs([Y(n/2+1:n); Y(1:n/2)]), 'r*');
legend('Orig. windowed sinusoid');
xlabel('Frequency (Hz)');
ylabel('Frequency response');
ylim([-1,max(amp)+1])

%pause;
%close;

%% Measurements
% measurement matrix phi, observation b
phi = randn(m,n);
b = phi*ywin;

% Overall measurement matrix; b = phi*F*F'*ywin, and F'*ywin is sparse.
A = phi*F;

x0 = A'*b/(m*n);
bigA = [real(A), -imag(A); imag(A), real(A)];
bigx0 = [real(x0); imag(x0)];
bigb = [real(b); imag(b)];


x0 = bigA'*bigb;

%% solve the LP
tic
bigYpcvx = l1eq_pd(x0, bigA, [], bigb, 10^(1));
time_l1 = toc;
Ypcvx = bigYpcvx(1:n) + 1i*bigYpcvx(n+1:2*n);
ypcvx = real(F*Ypcvx);


%% Solve l2 recovery

% Least-squares solution from b
tic
Ypmmse = (A'*inv(A*A'))*b;
time_l2 = toc;
ypmmse = real(F*Ypmmse);

% sub-sample and interpolate y
y_sub = y(1:floor(n/m):end);
length_y_sub = length(y_sub)
y_interp = interp(y_sub,floor(n/m));
y_interp_win = diag(hanning(length(y_interp)))*y_interp;


%% Solve OMP
s = min(10*k,m);
tic
Ypomp = OMP(A, b, s, []);
time_omp = toc;
ypomp = real(F*Ypomp);


%% Solve Reweighted L2  (Chartrand & Yin, ICASSP 2008)
%
%  Parameters:
%    p = 0.5  : lp exponent. Values in (0,1) give stronger sparsity
%               promotion than l1.  p = 0.5 is the standard choice
%               in Chartrand-Yin and balances sparsity vs. convergence speed.
%    eps_factor = 0.1 : aggressive geometric decrease of the smoothing
%               parameter eps.  Each outer iteration eps shrinks 10x,
%               so 30 iterations go from eps0 down to eps0 * 1e-30 (floored
%               at eps_min = 1e-8).  Faster decrease risks ill-conditioning
%               but converges in fewer iterations for well-conditioned A.
%    maxiter  = 30 : enough outer iterations for eps to reach its floor.
%
opts_rl2 = [];
opts_rl2.p          = 0.25;
opts_rl2.eps_factor = 1;
opts_rl2.eps_min    = 1e-8;
opts_rl2.maxiter    = 50;
opts_rl2.tol        = 1e-6;
opts_rl2.printEvery = 10;

fprintf('\n--- Reweighted L2 ---\n');
tic
Yprl2 = reweighted_l2(A, b, [], opts_rl2);
time_rl2 = toc;
yprl2 = real(F * Yprl2);


%% Solve Sparse Bayesian Learning (EM updates)  -- Algorithm 8, HW3 2026
%
%  Parameters:
%    sigma2 = 0.1*norm(b)^2/m : initial noise variance set to 10% of the
%               per-sample observation energy.  The EM will refine this.
%               For the noiseless experiment it drives quickly to near-zero.
%    update_sigma2 = true : let EM estimate sigma2.  Turning this off
%               (false) and fixing sigma2 to a small value can speed up
%               convergence when noiseless, but update=true is the general
%               principled choice.
%    maxiter = 500, tol = 1e-6 : SBL typically needs more iterations than
%               greedy methods but each iter is cheap once pruning kicks in.
%    prune_tol = 1e-10 : components whose posterior variance gamma_i falls
%               below this are set to zero and skipped.  This is the key
%               mechanism that gives SBL its sparsity.
%
opts_sbl = [];
% sigma2 left at default (1e-4 * norm(b)^2/M after col-normalisation inside sbl_em)
opts_sbl.update_sigma2  = false;
opts_sbl.maxiter        = 1000;
opts_sbl.tol            = 1e-12;
opts_sbl.prune_tol      = 9e-1;
opts_sbl.printEvery     = 100;

fprintf('\n--- Sparse Bayesian Learning (EM) ---\n');
tic
Ypsbl = sbl_em(A, b, [], opts_sbl);
time_sbl = toc;
ypsbl = real(F * Ypsbl);


%% Plots

% -- Frequency domain comparison --
figure;
freqAxis = ((0:1:n-1) - n/2) / 2;
fftshift_idx = [n/2+1:n, 1:n/2];

plot(freqAxis, abs(Y(fftshift_idx)),      'r*', ...
     freqAxis, abs(Ypcvx(fftshift_idx)),  'gd', ...
     freqAxis, abs(Ypmmse(fftshift_idx)), 'mx', ...
     freqAxis, abs(Ypomp(fftshift_idx)),  'kp', ...
     freqAxis, abs(Yprl2(fftshift_idx)),  'bs', ...
     freqAxis, abs(Ypsbl(fftshift_idx)),  'y^');
legend('Orig. windowed sinusoid', 'L1-min', 'L2-min', 'OMP', 'Rewtd-L2', 'SBL-EM');
xlabel('Frequency (Hz)');
ylabel('|Y(f)|');
title(['num. obsvns = ', num2str(m), ...
    ', num. vars = ', num2str(n), ', sparsity = ', num2str(k)]);
ylim([-1, max(amp)+1]);


% -- Time domain comparison --
figure;
tAxis     = (0:1:n-1) / n;
interpLen = length(y_interp_win);
tAxis_interp = (0:interpLen-1) / interpLen;
plot(tAxis, ywin(1:n),       'r-',  ...
     tAxis, ypcvx(1:n),      'g.-', ...
     tAxis, ypmmse(1:n),     'm--', ...
     tAxis_interp, y_interp_win, 'b:', ...
     tAxis, ypomp(1:n),      'k.', ...
     tAxis, yprl2(1:n),      'c-', ...
     tAxis, ypsbl(1:n),      'y-');
legend('Orig. windowed sinusoid', 'L1-min', 'L2-min', 'resample', 'OMP', 'Rewtd-L2', 'SBL-EM');
xlabel('Time');
ylabel('y(t)');
title(['num. obsvns = ', num2str(m), ...
    ', num. vars = ', num2str(n), ', sparsity = ', num2str(k)]);
ylim([-max(amp)-1, max(amp)+1]);


%% Performance summary table

mse_l1_freq   = sum(abs(Y - Ypcvx).^2)  / n;
mse_l2_freq   = sum(abs(Y - Ypmmse).^2) / n;
mse_omp_freq  = sum(abs(Y - Ypomp).^2)  / n;
mse_rl2_freq  = sum(abs(Y - Yprl2).^2)  / n;
mse_sbl_freq  = sum(abs(Y - Ypsbl).^2)  / n;

mse_l1_time   = sum(abs(ywin - ypcvx).^2)  / n;
mse_l2_time   = sum(abs(ywin - ypmmse).^2) / n;
mse_omp_time  = sum(abs(ywin - ypomp).^2)  / n;
mse_rl2_time  = sum(abs(ywin - yprl2).^2)  / n;
mse_sbl_time  = sum(abs(ywin - ypsbl).^2)  / n;

fprintf('\n=================================================================\n');
fprintf('%-15s  %10s  %12s  %12s\n', 'Algorithm', 'Time (s)', 'MSE (freq)', 'MSE (time)');
fprintf('-----------------------------------------------------------------\n');
fprintf('%-15s  %10.4f  %12.4e  %12.4e\n', 'L1-min',   time_l1,  mse_l1_freq,  mse_l1_time);
fprintf('%-15s  %10.4f  %12.4e  %12.4e\n', 'L2-min',   time_l2,  mse_l2_freq,  mse_l2_time);
fprintf('%-15s  %10.4f  %12.4e  %12.4e\n', 'OMP',      time_omp, mse_omp_freq, mse_omp_time);
fprintf('%-15s  %10.4f  %12.4e  %12.4e\n', 'Rewtd-L2', time_rl2, mse_rl2_freq, mse_rl2_time);
fprintf('%-15s  %10.4f  %12.4e  %12.4e\n', 'SBL-EM',   time_sbl, mse_sbl_freq, mse_sbl_time);
fprintf('=================================================================\n\n');

fprintf('Rewtd-L2 params : p=%.2f, eps_factor=%.2f, maxiter=%d\n', ...
        opts_rl2.p, opts_rl2.eps_factor, opts_rl2.maxiter);
fprintf('SBL-EM params   : sigma2_init=auto(1e-4*||b||^2/M), update_sigma2=%d, maxiter=%d, prune_tol=%.0e\n', ...
        opts_sbl.update_sigma2, opts_sbl.maxiter, opts_sbl.prune_tol);

