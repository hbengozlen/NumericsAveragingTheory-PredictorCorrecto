%% temporalConvergenceStudyPC.m
% Temporal (dt) convergence study for PredictorCorrector's splitting-scheme
% model (mainPC.m). Fixes the spatial grid and a total simulated time
% T_total, then refines dt repeatedly (nSubList sub-steps to cover the
% same T_total each time), comparing each run's final state against the
% finest-dt run to estimate the OBSERVED temporal order of accuracy.
%
% Expected order: ~2nd, since the scheme is predictor -> half-step BE ->
% velocity solve -> full-step CN corrector (see mainPC.m's Step 1-7) --
% unlike NumericsAveragingTheory's fully-coupled model, which is a plain
% backward-Euler step (~1st order; see
% NumericsAveragingTheory\temporalConvergenceStudy.m).
%
% A valid temporal study just needs ONE fixed theta0 -- unlike the spatial
% study (spatialConvergenceStudyPC.m), there's no cross-resolution
% consistency concern here, so getTheta0.m is called directly, same as
% mainPC.m itself.
%
% Self-contained: sets up its own grid/mesh/parameters (matching mainPC.m's
% own conventions/defaults) and calls the same solveXxxFFT step sequence
% mainPC.m's time loop uses (inlined below as the local function pcStep),
% instead of running mainPC.m itself -- no dependency on ModelComparison or
% NumericsAveragingTheory. See ModelComparison\compareModelsConvergence.m
% for the cross-model version of this idea (which also compares against
% the fully-coupled model). See spatialConvergenceStudyPC.m for the
% grid-refinement counterpart.

clear; clc;
% Hardcoded, not derived from mfilename('fullpath') -- when run from an
% editor that stages an unsaved/temp copy of the file, mfilename resolves
% to that temp buffer's path, not this file's actual saved location (bit
% NumericsAveragingTheory\temporalConvergenceStudy.m in exactly this way;
% see that file's header for the full story). Hardcoding sidesteps it here
% from the start rather than waiting to hit the same bug.
projRoot = 'Q:\Code Averaging Theory\Averaging Theroy Numerics\PredictorCorrector';
addpath(genpath(projRoot));

%% ===================== configuration =====================
% Physical/solver parameters -- matches mainPC.m's current defaults.
parameter.A=200e-6; parameter.etaf=1e-3; parameter.etab=1e-3;
parameter.R=0.7e-6; parameter.L=4e-6; parameter.b=4e-6; parameter.V0=5e-6;
parameter.ksif=-1; parameter.elasPara=36e-12; parameter.Db=1e-7*1e-4*20;
parameter.PrefPt=100; parameter.boolactive=1; parameter.boolpusher=0;
parameter.useRightPrecond=1;
parameter.gmresMaxIt = 100;
parameter.gmresRestartVelocity = 100;
parameter.gmresRestartScalar   = 30;
parameter.gmresTol=1e-15;   % single shared GMRES relative-residual target,
                            % read by all five *FFT solvers -- see
                            % mainPC.m's comment on this same field: not
                            % actually achievable at larger Nx (only
                            % ~1e-6-1e-7 reached at Nx=Ny=100 regardless of
                            % what's requested), which becomes a floor on
                            % how tight the temporal error measured below
                            % can meaningfully get before GMRES's own
                            % achievable precision starts to dominate it.

phi0 = 0.15;
Nx = 100; Ny = 100;
Theta0_cutoff = 3; Theta0_amplitude = 1;   % matches mainPC.m's current default
seed = 20260806;

T_total = 1e-4;              % matches mainPC.m's current dt*T ballpark (dt=1e-8, T=100*dt)
nSubList = [1 2 4 8 16 32 256 512 1024];   % dt = T_total/nSub -- order estimate below
                              % uses the actual ratio between whichever two
                              % dt's are compared, so this need not be a
                              % clean geometric sequence (same fix already
                              % applied in NumericsAveragingTheory's
                              % temporal/spatial studies).

%% ===================== grid/mesh setup =====================
parameter.Nx = Nx; parameter.Ny = Ny;
[parameter.x, Dx] = fourierGrid(1, Nx);
[parameter.y, Dy] = fourierGrid(1, Ny);
parameter.DX = kron(Dx, eye(Ny));
parameter.DY = kron(eye(Nx), Dy);
[~, D2x] = fourierGrid2(1, Nx);
[~, D2y] = fourierGrid2(1, Ny);
parameter.D2X = kron(D2x, eye(Ny));
parameter.D2Y = kron(eye(Nx), D2y);
D = fourierDiffOps(Nx, Ny);

%% ===================== initial condition =====================
rng(seed);
theta0 = getTheta0(Nx, Ny, Theta0_cutoff, Theta0_amplitude);
theta0 = theta0(:);

Ncoarse = (Nx-2)*(Ny-2); Nfine = Nx*Ny;
S0.phi   = phi0 * ones(Ncoarse, 1);
S0.theta = theta0;
[S0.Uf1, S0.Uf2, S0.Up1, S0.Up2, S0.P] = solveVelocityFFT(S0.phi, S0.theta, parameter, D);
%% ===================== run at each dt =====================
results = struct('nSub', {}, 'dt', {}, 'kFinal', {}, 'wallTime', {}, 'flagOk', {});
for i = 1:numel(nSubList)
    nSub = nSubList(i);
    dt = T_total / nSub;
    S = S0; Sprev = S0;   % Sprev=S0 on the first step -- degrades to no extrapolation, same as mainPC.m
    flagOk = true;
    tA = tic;
    for s = 1:nSub
        try
            [S, Sprev] = pcStep(S, Sprev, dt, parameter, D);
        catch ME
            fprintf('  FAILED at nSub=%d step=%d/%d: %s\n', nSub, s, nSub, ME.message);
            flagOk = false;
            break;
        end
    end
    wallTime = toc(tA);
    if flagOk
        kFinal = [S.phi; S.Uf1; S.Uf2; S.P; S.Up1; S.Up2; S.theta];
    else
        kFinal = NaN(2*Ncoarse + 5*Nfine - 1, 1);
    end
    results(i).nSub = nSub; results(i).dt = dt; results(i).kFinal = kFinal;
    results(i).wallTime = wallTime; results(i).flagOk = flagOk;
    fprintf('nSub=%2d  dt=%.3e  wall=%.3fs  flagOk=%d\n', nSub, dt, wallTime, flagOk);
end

%% ===================== analysis: self-convergence vs. the finest dt, Uf1 only =====================
% Error measured against THIS study's own finest-dt run, not pairwise
% between successive dt's -- same rationale as
% NumericsAveragingTheory\temporalConvergenceStudy.m's analysis section.
% Restricted to the Uf1 block (fluid velocity, x-component) only, not the
% whole 7-block state vector -- extractUf1.m isolates it using the same
% layout every solver in this project shares.
n = numel(results);
if ~results(n).flagOk
    warning('temporalConvergenceStudyPC:finestFailed', ...
        'The finest-dt run (nSub=%d) did not converge -- self-convergence numbers below are meaningless (comparing against a NaN reference).', nSubList(n));
end
finalRef = extractUf1(results(n).kFinal, Nx, Ny);
refNorm = norm(finalRef);
fprintf('\n--- Uf1 self-convergence vs finest dt=%.3e (expect ~2nd order); ||finestUf1||=%.4e ---\n', results(n).dt, refNorm);
d = nan(1, n-1);
for i = 1:n-1
    d(i) = norm(extractUf1(results(i).kFinal, Nx, Ny) - finalRef) / refNorm;
    fprintf('  dt=%.3e vs finest:  relative ||diff||=%.4e', results(i).dt, d(i));
    if i > 1
        r = results(i-1).dt / results(i).dt;   % actual refinement ratio for THIS pair
        fprintf('   ratio=%.3f  observed order p=%.2f', d(i-1)/d(i), log(d(i-1)/d(i))/log(r));
    end
    fprintf('\n');
end

%% ===================== plot =====================
dtList = [results(1:n-1).dt];
f = figure('Visible', 'off', 'Position', [100 100 800 550]);
loglog(dtList, d, 'o-', 'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', 'PredictorCorrector (splitting)');
hold on;
loglog(dtList, d(1)*(dtList/dtList(1)).^2, 'k--', 'DisplayName', 'slope 2 (reference)');
hold off;
set(gca, 'XDir', 'reverse');   % smaller dt (finer) to the right
xlabel('\Deltat'); ylabel('relative ||U^f_1(\Deltat) - U^f_1(finest \Deltat)||');
title(sprintf('PredictorCorrector (splitting) U^f_1 temporal self-convergence (relative), fixed N_x=N_y=%d', Nx));
legend('Location', 'southwest'); grid on;

outDir = fullfile(projRoot, 'plotting');
if ~exist(outDir, 'dir'); mkdir(outDir); end
outFile = fullfile(outDir, 'pc_temporal_convergence.png');
saveas(f, outFile);
close(f);
fprintf('\nSaved plot to %s\n', outFile);

%% ===================== wall time =====================
fprintf('\n--- wall time ---\n');
for i = 1:n
    fprintf('  nSub=%2d:  %.3fs\n', results(i).nSub, results(i).wallTime);
end

%% ===================== local functions =====================
function u = extractUf1(k, Nx, Ny)
% Isolates the Uf1 block (fluid velocity, x-component) from the full
% 7-block state vector layout [phi;Uf1;Uf2;P;Up1;Up2;theta] shared by
% every solver in this project (see pcStep's own S.phi/S.Uf1/... struct
% for the same layout, unconcatenated).
NP = (Nx-2)*(Ny-2); Nf = Nx*Ny;
u = k(NP+1 : NP+Nf);
end

function [S, Sprev] = pcStep(S, Sprev, dt_step, parameter, D)
% One predictor-corrector step -- exactly mainPC.m's Step 1-7 (time loop
% body), inlined here as a local function so both this script and
% spatialConvergenceStudyPC.m can call it without depending on mainPC.m
% itself (which is a top-level script, not a callable function).
%
% S:     current state (struct: phi, Uf1, Uf2, P, Up1, Up2, theta)
% Sprev: state one step before S -- used only for Step 4's extrapolated
%        GMRES warm-start (mainPC.m's Uf1_prev/etc.); returned updated to
%        S (pre-step) for the NEXT call, same bookkeeping as mainPC.m's
%        own Uf1_prev = Uf1_n line.

% Step 1: predictor -- reuse current velocities
Uf1_star = S.Uf1; Uf2_star = S.Uf2; Up1_star = S.Up1; Up2_star = S.Up2;

% Step 2: phi half-step BE
phi_half = solvePhiFFT(S.phi, Up1_star, Up2_star, dt_step/2, parameter, D);

% Step 3: theta half-step BE
theta_half = solveThetaFFT(S.theta, S.phi, Uf1_star, Uf2_star, Up1_star, Up2_star, dt_step/2, parameter, D);

% Step 4: velocity solve with half-step state (extrapolated warm-start)
x0_h = 1.5*[S.Uf1; S.Uf2; S.Up1; S.Up2; S.P] - 0.5*[Sprev.Uf1; Sprev.Uf2; Sprev.Up1; Sprev.Up2; Sprev.P];
[Uf1_h, Uf2_h, Up1_h, Up2_h, P_h] = solveVelocityFFT(phi_half, theta_half, parameter, D, x0_h);

% Step 5: phi full CN step
phi_n1 = solvePhiCNFFT(S.phi, Up1_h, Up2_h, dt_step, parameter, D);

% Step 6: theta full CN step
theta_n1 = solveThetaCNFFT(S.theta, phi_half, Uf1_h, Uf2_h, Up1_h, Up2_h, dt_step, parameter, D);

% Step 7: velocity solve at full step (extrapolated warm-start)
x0_n1 = 2*[Uf1_h; Uf2_h; Up1_h; Up2_h; P_h] - [S.Uf1; S.Uf2; S.Up1; S.Up2; S.P];
[Uf1_n1, Uf2_n1, Up1_n1, Up2_n1, P_n1] = solveVelocityFFT(phi_n1, theta_n1, parameter, D, x0_n1);

Sprev = S;   % this iteration's pre-step state becomes "prev" for the NEXT call
S.phi = phi_n1; S.theta = theta_n1;
S.Uf1 = Uf1_n1; S.Uf2 = Uf2_n1; S.Up1 = Up1_n1; S.Up2 = Up2_n1; S.P = P_n1;
end
