%% test_theta_fft_accuracy.m
% Compares solveTheta.m/solveThetaCN.m (dense, exact backslash) against
% solveThetaFFT.m/solveThetaCNFFT.m (matrix-free, GMRES + frozen-phibar/
% frozen-Up preconditioner) on the SAME theta_old/phi/Uf/Up, across a
% sweep of phi/Up spatial-variation amplitudes.
%
% Why sweep amplitude instead of a single value: the preconditioner
% freezes phi to its mean and Up1,Up2 to their means, so it's EXACT only
% when they're spatially uniform and degrades as their spatial variation
% grows -- same reasoning as solveVelocityFFT.m's/solvePhiFFT.m's frozen-
% coefficient preconditioners (see those tests' headers).
%
% solveThetaFFT.m/solveThetaCNFFT.m use RIGHT preconditioning (built by
% hand, since MATLAB's gmres only exposes LEFT preconditioner factors) for
% the same reason solveVelocityFFT.m does: its relres is the TRUE residual
% of the system, not a preconditioned stand-in.

clear; clc;
addpath('../theta', '../common');   % solveTheta.m, solveThetaFFT.m, fourierGrid*.m, etc.

Nx = 10; Ny = 10;
[x, Dx] = fourierGrid(1, Nx);
[xP, ~] = fourierGrid(1, Nx-2);
[y, Dy] = fourierGrid(1, Ny);
[yP, ~] = fourierGrid(1, Ny-2);
[~, D2x] = fourierGrid2(1, Nx);
[~, D2y] = fourierGrid2(1, Ny);
parameter.DX  = kron(Dx, eye(Ny));
parameter.DY  = kron(eye(Nx), Dy);
parameter.D2X = kron(D2x, eye(Ny));
parameter.D2Y = kron(eye(Nx), D2y);
parameter.Nx = Nx; parameter.Ny = Ny;
parameter.A        = 200e-6;
parameter.V0       = 5e-6;
parameter.etaf     = 1e-3;
parameter.R        = 0.7e-6;
parameter.L        = 4e-6;
parameter.elasPara = 36e-12;
parameter.ksif     = -1;
D = fourierDiffOps(Nx, Ny);

[xx, yy]   = meshgrid(x, y);   xx = xx(:);   yy = yy(:);
[xxP, yyP] = meshgrid(xP, yP); xxP = xxP(:); yyP = yyP(:);
[~, ~, CoarseMesh.interMatP] = getInterpolationMatrix(xxP, yyP, Nx-2, Ny-2, xx, yy, Nx, Ny);

Ncoarse = (Nx-2)*(Ny-2);
Nfine   = Nx*Ny;
dt_step = 1e-3;

rng(21);
theta_old = 0.05*randn(Nfine,1);
phi_base  = 0.15; rng(22); phi_noise = randn(Ncoarse,1);
Up1base = 0.3; Up2base = -0.2; Uf1base = 0.1; Uf2base = -0.15;
rng(23);
Up1noise = randn(Nfine,1); Up2noise = randn(Nfine,1);
Uf1noise = randn(Nfine,1); Uf2noise = randn(Nfine,1);

amps = [1e-6, 1e-3, 1e-1, 1, 5, 20];

%% Part 1: solveTheta.m (dense, exact) vs solveThetaFFT.m (matrix-free)
fprintf('=== Part 1: solveTheta.m (dense, exact) vs solveThetaFFT.m (matrix-free) ===\n\n');
results = zeros(numel(amps), 1);
for k = 1:numel(amps)
    a = amps(k);
    phi = phi_base + a*1e-3*phi_noise;   % phi_max<=1 physically, keep variation modest but still growing
    Up1 = Up1base + a*Up1noise; Up2 = Up2base + a*Up2noise;
    Uf1 = Uf1base + a*Uf1noise; Uf2 = Uf2base + a*Uf2noise;

    theta_dense = solveTheta(theta_old, phi, Uf1, Uf2, Up1, Up2, dt_step, parameter, CoarseMesh);
    theta_fft   = solveThetaFFT(theta_old, phi, Uf1, Uf2, Up1, Up2, dt_step, parameter, D);

    results(k) = max(abs(theta_dense - theta_fft)) / max(abs(theta_dense));
end

fprintf('\nRelative error (dense vs matrix-free), by amplitude:\n');
fprintf('%-12s %-10s\n', 'amp', 'theta err');
for k = 1:numel(amps)
    fprintf('%-12.3g %-10.3e\n', amps(k), results(k));
end

%% Part 2: solveThetaCN.m (dense, exact) vs solveThetaCNFFT.m (matrix-free)
fprintf('\n=== Part 2: solveThetaCN.m (dense, exact) vs solveThetaCNFFT.m (matrix-free) ===\n\n');
resultsCN = zeros(numel(amps), 1);
for k = 1:numel(amps)
    a = amps(k);
    phi = phi_base + a*1e-3*phi_noise;
    Up1 = Up1base + a*Up1noise; Up2 = Up2base + a*Up2noise;
    Uf1 = Uf1base + a*Uf1noise; Uf2 = Uf2base + a*Uf2noise;

    theta_dense = solveThetaCN(theta_old, phi, Uf1, Uf2, Up1, Up2, dt_step, parameter, CoarseMesh);
    theta_fft   = solveThetaCNFFT(theta_old, phi, Uf1, Uf2, Up1, Up2, dt_step, parameter, D);

    resultsCN(k) = max(abs(theta_dense - theta_fft)) / max(abs(theta_dense));
end

fprintf('\nRelative error (dense vs matrix-free), by amplitude:\n');
fprintf('%-12s %-10s\n', 'amp', 'theta err');
for k = 1:numel(amps)
    fprintf('%-12.3g %-10.3e\n', amps(k), resultsCN(k));
end
