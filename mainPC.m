clear all
clc
%% Parameters
parameter.A        = 200e-6;
parameter.etaf     = 1e-3;
parameter.etab     = 1e-3;
parameter.R        = 0.7e-6;
parameter.L        = 4e-6;
parameter.b        = 4e-6;
parameter.V0       = 5e-6;
parameter.ksif     = -1;
parameter.elasPara = 36e-12;
parameter.Db       = 10^(-7)*1e-4*20;
parameter.PrefPt   = 100;
parameter.boolactive = 1;
parameter.boolpusher = 0;
parameter.useFFT = 1;   % true: matrix-free FFT/GMRES solvers (solveVelocityFFT, solvePhiFFT); false: dense (solveVelocity, solvePhi)
parameter.useRightPrecond = 1;   % solveVelocityFFT.m's gmres: true=right preconditioning (trustworthy relres), false=left (kept for regression-testing only)
parameter.gmresTol = 1e-7;   % single shared GMRES relative-residual target, read by ALL five *FFT solvers
                              % (solveVelocityFFT, solvePhiFFT/solvePhiCNFFT, solveThetaFFT/solveThetaCNFFT)
                              % -- previously each hardcoded its own separate "tol = 1e-9", silently
                              % drifting out of sync if one was ever tuned without the others. NOTE:
                              % 1e-9 is untestable at larger grids -- GMRES simply cannot reach a
                              % relative residual that tight once the linear system is big enough
                              % (confirmed directly: only ~1e-6-1e-7 actually achieved at Nx=Ny=100,
                              % regardless of what's requested here), so a fixed value tuned for small
                              % grids silently becomes an unreachable target, wasting iterations chasing
                              % it and triggering "Input tol may not be achievable" warnings -- see
                              % ModelComparison's session notes if raising Nx/Ny substantially.
parameter.gmresMaxIt = 100;   % single shared GMRES outer-restart budget, read by ALL five *FFT
                              % solvers -- was previously a bare "100" hardcoded separately inside
                              % each gmres(...) call (same drift risk gmresTol had before it was
                              % centralized here).
parameter.gmresRestartVelocity = 100;   % solveVelocityFFT.m's GMRES restart (Krylov subspace size
                              % before each restart cycle) -- NOT the same value as gmresRestartScalar
                              % below: measured to matter a lot at Nx=Ny=40-100 (restart=30 forces
                              % GMRES to discard its whole Krylov subspace on every restart, costing
                              % 10-15x more total matvecs, or outright failing to converge within
                              % gmresMaxIt at Nx=Ny=100); solveVelocityFFT.m itself caps this at its
                              % own system size n=4*Nfine+Np, so this value only matters once n
                              % exceeds it.
parameter.gmresRestartScalar = 30;   % solvePhiFFT/solvePhiCNFFT/solveThetaFFT/solveThetaCNFFT's GMRES
                              % restart -- these scalar equations don't need velocity's larger
                              % Krylov subspace budget; kept as its own field (not unified with
                              % gmresRestartVelocity) since the two were independently tuned for
                              % different-sized/different-conditioned systems.

phi0             = 0.15;
t0               = 0;
dt               = 1e-4;
T                = 1;
Nx               = 30;
Ny               = 30;
Theta0_amplitude = 1e-3;
Theta0_cutoff    = 10;

parameter.Nx = Nx;
parameter.Ny = Ny;

%% Grid and differentiation matrices
addpath('velocity', 'phi', 'theta', 'common')   % per-solver-family subdirectories

[parameter.x, Dx] = fourierGrid(1, Nx);
[xP, DxP]           = fourierGrid(1, Nx-2);
[parameter.y, Dy] = fourierGrid(1, Ny);
[yP, DyP]           = fourierGrid(1, Ny-2);
parameter.DX = kron(Dx, eye(Ny));
parameter.DY = kron(eye(Nx), Dy);


% Direct 2nd-derivative matrices (fourierGrid2.m), NOT DX*DX/DY*DY -- DX
% zeros the Nyquist mode's derivative (ambiguous direction on an even
% grid), and DX*DX inherits that zero. A direct 2nd derivative has no such
% ambiguity (curvature is well-defined regardless of the Nyquist mode's
% assumed sign) and gives the correct nonzero value there instead. Used by
% solveVelocity.m's viscosity terms and solveTheta[CN].m's diffusion term.
[~, D2x] = fourierGrid2(1, Nx);
[~, D2y] = fourierGrid2(1, Ny);
parameter.D2X = kron(D2x, eye(Ny));
parameter.D2Y = kron(eye(Nx), D2y);

% Fourier eigenvalue-grid operators for solveVelocityFFT.m's matrix-free
% path -- computed once here (Nx,Ny are fixed for the whole run) and
% passed down explicitly, rather than solveVelocityFFT caching it itself.
D = fourierDiffOps(Nx, Ny);

[xx, yy]   = meshgrid(parameter.x, parameter.y);
xx = xx(:); yy = yy(:);
[xxP, yyP] = meshgrid(xP, yP);
xxP = xxP(:); yyP = yyP(:);

FineMesh.xx  = xx;    FineMesh.yy  = yy;
CoarseMesh.xx = xxP;  CoarseMesh.yy = yyP;

[~, ~, CoarseMesh.interMatP] = getInterpolationMatrix(xxP, yyP, Nx-2, Ny-2, xx,  yy,  Nx,   Ny);
% [~, ~, FineMesh.interMatU]   = getInterpolationMatrix(xx,  yy,  Nx,   Ny,   xxP, yyP, Nx-2, Ny-2);
FineMesh.interMatU = buildRestrictTruncMat(Nx, Ny);
%% Initial conditions  (boolpsi=0: phi directly on coarse grid)
Ncoarse = (Nx-2)*(Ny-2);
Nfine   = Nx*Ny;

phi_n   = phi0 * ones(Ncoarse, 1);
phi_n   = phi_n(:);
theta_n = getTheta0(Nx, Ny, Theta0_cutoff, Theta0_amplitude);
theta_n=theta_n(:);
[Uf1_n, Uf2_n, Up1_n, Up2_n, P_n] =  solveVelocityFFT(phi_n, theta_n, parameter, D);

% Uf1_n   = zeros(Nfine, 1);
% Uf2_n   = zeros(Nfine, 1);
% P_n     = parameter.PrefPt * ones(Ncoarse-1, 1);
% Up1_n   = zeros(Nfine, 1);
% Up2_n   = zeros(Nfine, 1);

% Retained across iterations only for Step 4's extrapolated warm-start
% below (see that comment) -- initialized equal to the n=0 state, so the
% very first iteration's extrapolation degrades gracefully to "no
% extrapolation" (same as just reusing Uf1_n, etc., as before).
Uf1_prev = Uf1_n; Uf2_prev = Uf2_n; Up1_prev = Up1_n; Up2_prev = Up2_n; P_prev = P_n;

%% Storage
t   = t0;
tt  = t0;
kk  = [phi_n; Uf1_n; Uf2_n; P_n; Up1_n; Up2_n; theta_n];

%% Profiling accumulators
% solveVelocity/solveVelocityFFT's internal timing lives in perfCounter's
% own persistent state (no global variable) -- reset here, read back via
% perfCounter(...,'get') in the Timing report section below.
perfCounter('solveVelocity', 'reset');
perfCounter('solveVelocityFFT', 'reset');
timing = struct('solvePhi', 0, 'solveTheta', 0, 'solveVelocity', 0, ...
                 'solvePhiCN', 0, 'solveThetaCN', 0, 'nsteps', 0);
runTic = tic;
useFFT=parameter.useFFT;
%% Time loop
while t < T
    stepWallTic = tic;
    dt_step = min(dt, T - t);

    % Step 1: predictor — reuse current velocities
    Uf1_star = Uf1_n;
    Uf2_star = Uf2_n;
    Up1_star = Up1_n;
    Up2_star = Up2_n;

    % Step 2: phi half-step BE
    stepTic = tic;
    if useFFT
        phi_half = solvePhiFFT(phi_n, Up1_star, Up2_star, dt_step/2, parameter, D);
    else
        phi_half = solvePhi(phi_n, Up1_star, Up2_star, dt_step/2, parameter, FineMesh, CoarseMesh);
    end
    timing.solvePhi = timing.solvePhi + toc(stepTic);

    % Step 3: theta half-step BE
    stepTic = tic;
    if useFFT
        theta_half = solveThetaFFT(theta_n, phi_n, Uf1_star, Uf2_star, Up1_star, Up2_star, dt_step/2, parameter, D);
    else
        theta_half = solveTheta(theta_n, phi_n, Uf1_star, Uf2_star, Up1_star, Up2_star, dt_step/2, parameter, CoarseMesh);
    end
    timing.solveTheta = timing.solveTheta + toc(stepTic);

    % Step 4: velocity solve with half-step state
    stepTic = tic;
    if useFFT
        % Warm-start via linear extrapolation across the last outer step
        % (Uf1_prev at t-dt_prev, Uf1_n at t) forward by half THIS step,
        % instead of just reusing Uf1_n unchanged -- a systematically
        % better guess whenever the solution is actually drifting step to
        % step (the exact regime where plain reuse needs more GMRES
        % iterations: iters=10/8 cold vs iters=2/23 warm-started with the
        % TRUE solution, verified earlier -- extrapolation aims to land
        % closer to that true solution than last step's converged value
        % alone). Reduces to x0_h=[Uf1_n;...] (previous behavior) on the
        % first iteration, since Uf1_prev=Uf1_n there.
        x0_h = 1.5*[Uf1_n; Uf2_n; Up1_n; Up2_n; P_n] ...
             - 0.5*[Uf1_prev; Uf2_prev; Up1_prev; Up2_prev; P_prev];
        [Uf1_h, Uf2_h, Up1_h, Up2_h, P_h] = solveVelocityFFT(phi_half, theta_half, parameter, D, x0_h);
    else
        [Uf1_h, Uf2_h, Up1_h, Up2_h, P_h] = solveVelocity(phi_half, theta_half, parameter, FineMesh, CoarseMesh);
    end
    timing.solveVelocity = timing.solveVelocity + toc(stepTic);

    % Step 5: phi full CN step
    stepTic = tic;
    if useFFT
        phi_n1 = solvePhiCNFFT(phi_n, Up1_h, Up2_h, dt_step, parameter, D);
    else
        phi_n1 = solvePhiCN(phi_n, Up1_h, Up2_h, dt_step, parameter, FineMesh, CoarseMesh);
    end
    timing.solvePhiCN = timing.solvePhiCN + toc(stepTic);

    % Step 6: theta full CN step
    stepTic = tic;
    if useFFT
        theta_n1 = solveThetaCNFFT(theta_n, phi_half, Uf1_h, Uf2_h, Up1_h, Up2_h, dt_step, parameter, D);
    else
        theta_n1 = solveThetaCN(theta_n, phi_half, Uf1_h, Uf2_h, Up1_h, Up2_h, dt_step, parameter, CoarseMesh);
    end
    timing.solveThetaCN = timing.solveThetaCN + toc(stepTic);

    % Step 7: velocity solve at full step to get consistent u^{n+1}, P^{n+1}
    stepTic = tic;
    if useFFT
        % Warm-start via linear extrapolation from the two points already
        % in hand THIS iteration -- Uf1_n at t, Uf1_h at t+dt_step/2 --
        % forward by another half step to t+dt_step, instead of just
        % reusing Uf1_h unchanged. No extra retained state needed (unlike
        % Step 4's), since both points come from this same iteration.
        x0_n1 = 2*[Uf1_h; Uf2_h; Up1_h; Up2_h; P_h] - [Uf1_n; Uf2_n; Up1_n; Up2_n; P_n];
        [Uf1_n1, Uf2_n1, Up1_n1, Up2_n1, P_n1] = solveVelocityFFT(phi_n1, theta_n1, parameter, D, x0_n1);
    else
        [Uf1_n1, Uf2_n1, Up1_n1, Up2_n1, P_n1] = solveVelocity(phi_n1, theta_n1, parameter, FineMesh, CoarseMesh);
    end
    timing.solveVelocity = timing.solveVelocity + toc(stepTic);

    timing.nsteps = timing.nsteps + 1;

    % Save this step's (about-to-be-overwritten) state as "prev", for the
    % NEXT iteration's Step 4 extrapolated warm-start above.
    Uf1_prev = Uf1_n; Uf2_prev = Uf2_n; Up1_prev = Up1_n; Up2_prev = Up2_n; P_prev = P_n;

    % Advance state
    phi_n   = phi_n1;
    theta_n = theta_n1;
    Uf1_n = Uf1_n1;  Uf2_n = Uf2_n1;
    Up1_n = Up1_n1;  Up2_n = Up2_n1;
    P_n   = P_n1;

    t   = t + dt_step;
    tt  = [tt, t];
    kk  = [kk, [phi_n; Uf1_n; Uf2_n; P_n; Up1_n; Up2_n; theta_n]];

    phiext_now = CoarseMesh.interMatP * phi_n;
    fprintf('t=%.4f  phi_mean=%.4f  phi_min=%.4f  phi_max=%.4f  [step wall time: %.3f s]\n', ...
        t, mean(phiext_now), min(phiext_now), max(phiext_now), toc(stepWallTic));
    if min(phiext_now) < 0
    fprintf('phi_min=%.4e < 0, stopping simulation.\n', min(phiext_now));
    break;
    end
end

%% Timing report
totalTime = toc(runTic);
fprintf('\n=== Timing summary over %d steps (%.3f s total) ===\n', timing.nsteps, totalTime);
fprintf('%-14s %8.3f s  (%5.1f%%)\n', 'solvePhi',     timing.solvePhi,     100*timing.solvePhi/totalTime);
fprintf('%-14s %8.3f s  (%5.1f%%)\n', 'solveTheta',   timing.solveTheta,   100*timing.solveTheta/totalTime);
fprintf('%-14s %8.3f s  (%5.1f%%)\n', 'solveVelocity',timing.solveVelocity,100*timing.solveVelocity/totalTime);
fprintf('%-14s %8.3f s  (%5.1f%%)\n', 'solvePhiCN',   timing.solvePhiCN,   100*timing.solvePhiCN/totalTime);
fprintf('%-14s %8.3f s  (%5.1f%%)\n', 'solveThetaCN', timing.solveThetaCN, 100*timing.solveThetaCN/totalTime);
velocityTiming = perfCounter('solveVelocity', 'get', {'build','solve','calls'});
fprintf('\n--- solveVelocity breakdown (%d calls) ---\n', velocityTiming.calls);
fprintf('%-14s %8.3f s  (%5.1f%% of solveVelocity)\n', 'build matrix', velocityTiming.build, 100*velocityTiming.build/timing.solveVelocity);
fprintf('%-14s %8.3f s  (%5.1f%% of solveVelocity)\n', 'A\\rhs solve',  velocityTiming.solve, 100*velocityTiming.solve/timing.solveVelocity);

velocityFFTTiming = perfCounter('solveVelocityFFT', 'get', {'setup','Av','Pinv','calls','AvCalls','PinvCalls'});
fprintf('\n--- solveVelocityFFT breakdown (%d calls, %d Av calls, %d Pinv calls) ---\n', ...
    velocityFFTTiming.calls, velocityFFTTiming.AvCalls, velocityFFTTiming.PinvCalls);
gmresOverhead = timing.solveVelocity - velocityFFTTiming.setup - velocityFFTTiming.Av - velocityFFTTiming.Pinv;
fprintf('%-22s %8.3f s  (%5.1f%% of solveVelocity)\n', 'setup (pre-gmres)', velocityFFTTiming.setup, 100*velocityFFTTiming.setup/timing.solveVelocity);
fprintf('%-22s %8.3f s  (%5.1f%% of solveVelocity)\n', 'Av total',          velocityFFTTiming.Av,    100*velocityFFTTiming.Av/timing.solveVelocity);
fprintf('%-22s %8.3f s  (%5.1f%% of solveVelocity)\n', 'Pinv total',        velocityFFTTiming.Pinv,  100*velocityFFTTiming.Pinv/timing.solveVelocity);
fprintf('%-22s %8.3f s  (%5.1f%% of solveVelocity)\n', 'gmres overhead',    gmresOverhead,           100*gmresOverhead/timing.solveVelocity);
