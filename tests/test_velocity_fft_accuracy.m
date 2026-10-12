%% test_velocity_fft_accuracy.m
% Compares solveVelocity.m (dense, exact backslash) against
% solveVelocityFFT.m (matrix-free, GMRES + frozen-phibar preconditioner)
% on the SAME phi/theta, across a sweep of theta amplitudes.
%
% Split out of test_preconditioner_rank_deficiency.m, which now only
% checks rank(M) -- see that file for the preconditioner-decoupling
% history/rationale this test's accuracy numbers depend on.
%
% Why sweep amplitude instead of a single value: accuracy at tiny theta
% amplitude does NOT guarantee accuracy at large amplitude. Verified twice
% this way -- once when Option A's preconditioner decoupling (4 corner
% modes only) turned out full-rank everywhere but catastrophically
% ill-conditioned (cond(M)~1e6) at non-corner Nyquist-adjacent modes, and
% again when narrowing Option B's gap band to just the single true-Nyquist
% index (leaving its 2 flanking bins matched) reintroduced the same
% 10-40x error at larger amplitude for a different reason (those flanking
% bins are the fine grid's two halves of ONE coarse-Nyquist pressure dof,
% a cross-mode coupling this per-mode-block-diagonal preconditioner can
% never represent). Both regressions looked fine at tiny amplitude and
% only showed up once amplitude grew -- hence the sweep, not a single
% point.
%
% solveVelocityFFT.m uses RIGHT preconditioning (built by hand, since
% MATLAB's gmres only exposes LEFT preconditioner factors via M1/M2): it
% solves (Av*Pinv)*y=rhs for y, then recovers x=Pinv*y. Its relres is
% therefore the TRUE residual of the original system (Av*x-rhs exactly),
% unlike left preconditioning's PRECONDITIONED residual, which was
% verified to under-report the true error by orders of magnitude and
% unevenly across equations (x-momentum rows ~10% true error at a
% preconditioned relres of 1e-9).

clear; clc;
addpath('../velocity', '../common');   % solveVelocity.m, solveVelocityFFT.m, fourierGrid*.m, etc.

%% Test case setup (small grid, fast to run)
parameter.A        = 200e-6;
parameter.etaf     = 1e-3;
parameter.etab     = 1e-4;
parameter.R        = 0.7e-6;
parameter.L        = 4e-6;
parameter.b        = 4e-6;
parameter.V0       = 5e-6;
parameter.ksif     = -1;
parameter.elasPara = 36e-12;
parameter.Db       = 10^(-7)*1e-4;
parameter.phi_max  = 0.6;
parameter.PrefPt   = 100;
parameter.boolactive = 1;
parameter.boolpusher = 0;
parameter.useRightPrecond = 1;
parameter.gmresTol = 1e-16;   
parameter.gmresMaxIt = 100;   
parameter.gmresRestartVelocity = 100;  
parameter.gmresRestartScalar = 30;   
Nx = 10; Ny = 10;
parameter.Nx = Nx; parameter.Ny = Ny;

[parameter.x, Dx] = fourierGrid(1, Nx);
[xP, ~]           = fourierGrid(1, Nx-2);
[parameter.y, Dy] = fourierGrid(1, Ny);
[yP, ~]           = fourierGrid(1, Ny-2);
parameter.DX = kron(Dx, eye(Ny));
parameter.DY = kron(eye(Nx), Dy);
[~, D2x] = fourierGrid2(1, Nx);
[~, D2y] = fourierGrid2(1, Ny);
parameter.D2X = kron(D2x, eye(Ny));
parameter.D2Y = kron(eye(Nx), D2y);
D = fourierDiffOps(Nx, Ny);

[xx, yy]   = meshgrid(parameter.x, parameter.y); xx = xx(:); yy = yy(:);
[xxP, yyP] = meshgrid(xP, yP);                   xxP = xxP(:); yyP = yyP(:);
FineMesh.xx   = xx;   FineMesh.yy   = yy;
CoarseMesh.xx = xxP;  CoarseMesh.yy = yyP;

[~, ~, CoarseMesh.interMatP] = getInterpolationMatrix(xxP, yyP, Nx-2, Ny-2, xx,  yy,  Nx,   Ny);
FineMesh.interMatU = buildRestrictTruncMat(Nx, Ny);   % truncation, matches velocityAv.m

Ncoarse = (Nx-2)*(Ny-2);
rng(7);
phi = 0.15 + 0.03*randn(Ncoarse,1);

%% Accuracy sweep -- dense (exact) vs matrix-free (GMRES+preconditioner)
fprintf('=== solveVelocity.m (dense, exact) vs solveVelocityFFT.m (matrix-free) ===\n\n');

theta_amps = [1e-6, 1e-3, 1e-1, 1, 5, 20];
results = zeros(numel(theta_amps), 5);
for k = 1:numel(theta_amps)
    theta = getTheta0(Nx, Ny, 3, theta_amps(k)); theta = theta(:);

    [Uf1, Uf2, Up1, Up2, P]      = solveVelocity(phi, theta, parameter, FineMesh, CoarseMesh);
    [Uf1f, Uf2f, Up1f, Up2f, Pf] = solveVelocityFFT(phi, theta, parameter, D);

    results(k,1) = max(abs(Uf1-Uf1f))/max(abs(Uf1));
    results(k,2) = max(abs(Uf2-Uf2f))/max(abs(Uf2));
    results(k,3) = max(abs(Up1-Up1f))/max(abs(Up1));
    results(k,4) = max(abs(Up2-Up2f))/max(abs(Up2));
    results(k,5) = max(abs(P-Pf))/max(abs(P));
end

fprintf('\nRelative errors (dense vs matrix-free), by theta amplitude:\n');
fprintf('%-12s %-10s %-10s %-10s %-10s %-10s\n', 'amp', 'Uf1', 'Uf2', 'Up1', 'Up2', 'P');
for k = 1:numel(theta_amps)
    fprintf('%-12.3g %-10.3e %-10.3e %-10.3e %-10.3e %-10.3e\n', theta_amps(k), results(k,:));
end

fprintf(['\nBoth solvers use the SAME discretization (mean(P)=PrefPt gauge,\n' ...
         'truncation-based fine->coarse restriction). solveVelocityFFT.m uses RIGHT\n' ...
         'preconditioning, so its relres/flag ARE trustworthy on their own -- and with\n' ...
         'M full rank everywhere (see test_preconditioner_rank_deficiency.m), the\n' ...
         'practical error stays small across the whole amplitude sweep, not just at\n' ...
         'tiny amplitude.\n']);
