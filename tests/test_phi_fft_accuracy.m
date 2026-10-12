%% test_phi_fft_accuracy.m
% Compares solvePhi.m/solvePhiCN.m (dense, exact backslash) against
% solvePhiFFT.m/solvePhiCNFFT.m (matrix-free, GMRES + frozen-mean-Up
% preconditioner) on the SAME phi_old/Up1/Up2, across a sweep of Up
% spatial-variation amplitudes.
%
% Why sweep amplitude instead of a single value: the preconditioner
% freezes Up1,Up2 to their spatial means, so it's EXACT only when Up is
% spatially uniform and degrades as Up's spatial variation grows -- same
% reasoning as solveVelocityFFT.m's frozen-phibar preconditioner, and the
% same reason THAT test sweeps theta amplitude instead of checking one
% point (see test_velocity_fft_accuracy.m's header).
%
% solvePhiFFT.m/solvePhiCNFFT.m use RIGHT preconditioning (built by hand,
% since MATLAB's gmres only exposes LEFT preconditioner factors) for the
% same reason solveVelocityFFT.m does: its relres is the TRUE residual of
% the system, not a preconditioned stand-in.

clear; clc;
addpath('../phi', '../common');   % solvePhi.m, solvePhiFFT.m, fourierGrid*.m, etc.

Nx = 10; Ny = 10;
[x, Dx] = fourierGrid(1, Nx);
[xP, ~] = fourierGrid(1, Nx-2);
[y, Dy] = fourierGrid(1, Ny);
[yP, ~] = fourierGrid(1, Ny-2);
parameter.DX = kron(Dx, eye(Ny));
parameter.DY = kron(eye(Nx), Dy);
parameter.Nx = Nx; parameter.Ny = Ny;
D = fourierDiffOps(Nx, Ny);

[xx, yy]   = meshgrid(x, y);   xx = xx(:);   yy = yy(:);
[xxP, yyP] = meshgrid(xP, yP); xxP = xxP(:); yyP = yyP(:);
FineMesh.xx = xx; FineMesh.yy = yy;
FineMesh.interMatU = buildRestrictTruncMat(Nx, Ny);   % truncation, matches phiAv.m
[~, ~, CoarseMesh.interMatP] = getInterpolationMatrix(xxP, yyP, Nx-2, Ny-2, xx, yy, Nx, Ny);

Ncoarse = (Nx-2)*(Ny-2);
Nfine   = Nx*Ny;
dt_step = 1e-3;

rng(11);
phi_old = 0.15 + 0.03*randn(Ncoarse,1);

%% Sweep the amplitude of Up's spatial variation around a fixed nonzero mean
Up_amps = [1e-6, 1e-3, 1e-1, 1, 5, 20];
results = zeros(numel(Up_amps), 1);

Up1base = 0.3; Up2base = -0.2;   % nonzero means, so the frozen-mean preconditioner has something nontrivial to freeze to
rng(12);
Up1noise = randn(Nfine,1);
Up2noise = randn(Nfine,1);

fprintf('=== Part 1: solvePhi.m (dense, exact) vs solvePhiFFT.m (matrix-free) ===\n\n');
for k = 1:numel(Up_amps)
    Up1 = Up1base + Up_amps(k)*Up1noise;
    Up2 = Up2base + Up_amps(k)*Up2noise;

    phi_dense = solvePhi(phi_old, Up1, Up2, dt_step, parameter, FineMesh, CoarseMesh);
    phi_fft   = solvePhiFFT(phi_old, Up1, Up2, dt_step, parameter, D);

    results(k) = max(abs(phi_dense - phi_fft)) / max(abs(phi_dense));
end

fprintf('\nRelative error (dense vs matrix-free), by Up spatial-variation amplitude:\n');
fprintf('%-12s %-10s\n', 'amp', 'phi err');
for k = 1:numel(Up_amps)
    fprintf('%-12.3g %-10.3e\n', Up_amps(k), results(k));
end

%% Part 2: solvePhiCN.m (dense, exact) vs solvePhiCNFFT.m (matrix-free)
fprintf('\n=== Part 2: solvePhiCN.m (dense, exact) vs solvePhiCNFFT.m (matrix-free) ===\n\n');
resultsCN = zeros(numel(Up_amps), 1);
for k = 1:numel(Up_amps)
    Up1 = Up1base + Up_amps(k)*Up1noise;
    Up2 = Up2base + Up_amps(k)*Up2noise;

    phi_dense = solvePhiCN(phi_old, Up1, Up2, dt_step, parameter, FineMesh, CoarseMesh);
    phi_fft   = solvePhiCNFFT(phi_old, Up1, Up2, dt_step, parameter, D);

    resultsCN(k) = max(abs(phi_dense - phi_fft)) / max(abs(phi_dense));
end

fprintf('\nRelative error (dense vs matrix-free), by Up spatial-variation amplitude:\n');
fprintf('%-12s %-10s\n', 'amp', 'phi err');
for k = 1:numel(Up_amps)
    fprintf('%-12.3g %-10.3e\n', Up_amps(k), resultsCN(k));
end
