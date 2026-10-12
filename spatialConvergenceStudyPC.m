%% spatialConvergenceStudyPC.m
% Spatial (Nx=Ny) convergence study for PredictorCorrector's splitting-
% scheme model (mainPC.m), for MULTIPLE fixed dt's -- fixes dt and the
% number of time steps, then refines the grid repeatedly, comparing each
% resolution's Uf1 (fluid velocity, x-component) against THAT SAME dt's
% own finest-Nx run's Uf1 (all resampled onto that finest grid) to
% estimate the OBSERVED spatial order of accuracy -- repeated once per
% entry in dtList_s, so a single plot shows one curve per dt, each self-
% referenced against its own finest-Nx run (not a single shared reference
% across dt's). Restricted to Uf1 only (not the whole 7-block state
% vector) via the local function extractUf1.
%
% Why multiple dt's: this solver's Uf1 solve at any given step is a
% direct elliptic solve, not iterated over dt -- so in principle the Uf1
% spatial-error curve should be dt-INDEPENDENT (unlike phi/theta, which
% get one BE + one CN update per step and so do carry a genuine temporal
% error). Running the same Nx sweep at several dt's lets you actually
% check that, rather than assume it: if the curves overlap, Uf1's spatial
% error really is dt-independent here; if coarser dt's sit above finer
% ones, temporal error is leaking into what's nominally a pure spatial-
% refinement measurement (e.g. via phi_half/theta_half, which Uf1's own
% solve depends on and which DO carry dt-dependent error).
%
% Expected order: like NumericsAveragingTheory's fully-coupled model (see
% that codebase's spatialConvergenceStudy.m), this solver's spatial
% derivatives (DX/DY/D2X/D2Y, built from the same fourierGrid/fourierGrid2
% convention) are Fourier SPECTRAL, and getInterpolationMatrix.m's
% coarse<->fine coupling (trigonometric collocation) is spectral too, not
% a fixed-order interpolation stencil. So spatial error here is also
% expected to decay near-exponentially with Nx, not at a fixed polynomial
% order the way PC's TIME discretization is (predictor-corrector is
% 2nd-order in dt, fixed, regardless of grid -- see
% temporalConvergenceStudyPC.m). If the plot below deviates from a
% straight line at coarse Nx, that's most likely gmresTol's own achievable-
% precision floor (see mainPC.m's comment on that field) starting to
% dominate, not genuinely non-spectral behavior.
%
% A valid spatial study needs the SAME underlying continuous theta0 field
% at every tested resolution -- calling getTheta0.m independently at each
% Nx draws independent randomness per resolution (the surviving low-
% frequency coefficients land at different array indices for different
% Nx), which would silently change the problem being solved at each grid
% size instead of just how well it's resolved. The local helper functions
% below (localBuildConsistentTheta0Seed / localMaterializeTheta0) sidestep
% this the same way NumericsAveragingTheory\spatialConvergenceStudy.m /
% ModelComparison\buildConsistentTheta0Seed.m do: draw the randomness ONCE
% at a small reference resolution, then spectrally upsample that SAME
% field to whatever target (Nx,Ny) is needed. Comparing different-
% resolution results against each other also needs them on a common grid
% -- localResampleState does that, spectrally upsampling every coarser
% result onto the finest tested grid before differencing.
%
% Self-contained: sets up its own grid/mesh/parameters (matching
% mainPC.m's own conventions/defaults) and calls the same solveXxxFFT step
% sequence mainPC.m's time loop uses (inlined below as the local function
% pcStep) -- no dependency on ModelComparison or NumericsAveragingTheory
% (the local helper functions below are self-contained copies of the same
% validated logic those use, not a path dependency on them). See
% ModelComparison\compareModelsConvergence.m for the cross-model version of
% this idea. See temporalConvergenceStudyPC.m for the dt-refinement
% counterpart.

clear; clc;
% Hardcoded, not derived from mfilename('fullpath') -- see
% temporalConvergenceStudyPC.m's header for why.
projRoot = 'Q:\Code Averaging Theory\Averaging Theroy Numerics\PredictorCorrector';
addpath(genpath(projRoot));

%% ===================== configuration =====================
% Physical/solver parameters -- matches mainPC.m's current defaults.
parameter.A=200e-6; parameter.etaf=1e-3; parameter.etab=1e-3;
parameter.R=0.7e-6; parameter.L=4e-6; parameter.b=4e-6; parameter.V0=5e-6;
parameter.ksif=-1; parameter.elasPara=36e-12; parameter.Db=1e-7*1e-4*20;
parameter.PrefPt=100; parameter.boolactive=1; parameter.boolpusher=0;
parameter.useRightPrecond=1;
parameter.gmresMaxIt = 200;
parameter.gmresRestartVelocity = 100;
parameter.gmresRestartScalar   = 30;
parameter.useFFT = 1;   % true (default): matrix-free FFT/GMRES solvers (solveVelocityFFT,
                        % solvePhiFFT, etc.), same spectral solvers this study normally
                        % exercises. false: mainPC.m's dense/direct solvers (solveVelocity,
                        % solvePhi, solveTheta, ...) instead -- lets you check whether the
                        % dt-dependent behavior seen in the FFT/GMRES path (accumulated
                        % GMRES-floor noise growing with step count, see the multi-dt plot
                        % this script produces) is specific to the iterative solves, or
                        % shows up with the dense direct path too. When false, an extra
                        % FineMesh/CoarseMesh pair (mirrors mainPC.m's own setup) is built
                        % per Nx below, and pcStep dispatches to the non-FFT solver family.
parameter.gmresTol=1e-13;   % single shared GMRES relative-residual target
                            % -- see temporalConvergenceStudyPC.m's/
                            % mainPC.m's comment on this field: not
                            % actually achievable at larger Nx, which
                            % becomes a floor on how tight the spatial
                            % error measured below can meaningfully get.

phi0 = 0.15;
Theta0_cutoff = 3; Theta0_amplitude = 1;   % matches mainPC.m's current default

seed = 20260806;

dtList_s = [1e-5];   % multiple dt's -- see header. Narrowed
                     % from a wider range that included 1e-8/1e-9: with
                     % T_total_s now FIXED (see below), holding T_total
                     % fixed across dt's spanning 5 decades would need
                     % anywhere from ~1 to ~1e6 steps depending on dt,
                     % which isn't practical -- 2 decades keeps nSteps
                     % (10 to 1000 here) reasonable for every entry.
T_total_s = 20e-5;    % FIXED across every dt -- nSteps_s alone (previously
                     % fixed regardless of dt) let T_total = nSteps_s*dt
                     % shrink proportionally with dt, so smaller dt's
                     % simulated less real physical time and produced a
                     % smaller-magnitude Uf1 signal against the same fixed
                     % absolute GMRES/discretization noise floor --
                     % inflating RELATIVE error at small dt (verified:
                     % dt=1e-8/1e-9 showed WORSE self-convergence than
                     % dt=1e-5, the opposite of what refining dt should
                     % do). Fixing T_total instead (nSteps = round(T_total_s/dt),
                     % like temporalConvergenceStudyPC.m already does)
                     % keeps every dt simulating the SAME physical time, so
                     % Uf1's magnitude stays comparable across the sweep
                     % and the relative-error comparison actually isolates
                     % spatial error.

NxList_s = [10 20 30 40 50 60 70 80 90 100 120];   % same list used by
                     % NumericsAveragingTheory\spatialConvergenceStudy.m,
                     % for easy side-by-side comparison of the two models'
                     % spatial behavior. Need not be a constant ratio --
                     % the order estimate below uses the ACTUAL ratio
                     % between whichever two Nx's are being compared.

%% ===================== resolution-independent theta0 seed =====================
coeffs = localBuildConsistentTheta0Seed(Theta0_cutoff, seed);

%% ===================== run at each (dt, Nx), analyze per dt =====================
nDt = numel(dtList_s);
nNx = numel(NxList_s);
dAll = nan(nDt, nNx-1);   % relative Uf1 error, one row per dt, vs THAT dt's own finest-Nx run
wallAll = nan(nDt, nNx);

for dtIdx = 1:nDt
    dt_s = dtList_s(dtIdx);
    nSteps_s = max(1, round(T_total_s / dt_s));   % same T_total_s for every dt -- see config comment above
    fprintf('\n########## dt_s=%.3e (nSteps=%d, T_total=%.3e) ##########\n', dt_s, nSteps_s, nSteps_s*dt_s);

    results = struct('Nx', {}, 'Ny', {}, 'kFinal', {}, 'wallTime', {}, 'flagOk', {});
    for i = 1:nNx
        Nx = NxList_s(i); Ny = Nx;
        fprintf('-- Nx=Ny=%d --\n', Nx);

        %% grid/mesh setup (depends on Nx,Ny -- rebuilt every iteration)
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

        % Dense-path mesh (FineMesh/CoarseMesh) -- only needed when
        % useFFT=0, but building it is cheap relative to the solves
        % below and mirrors mainPC.m's own setup exactly (see its header
        % comment for what interMatP/interMatU are for).
        FineMesh = []; CoarseMesh = [];
        if ~parameter.useFFT
            [xP, ~] = fourierGrid(1, Nx-2);
            [yP, ~] = fourierGrid(1, Ny-2);
            [xx, yy]   = meshgrid(parameter.x, parameter.y);
            xx = xx(:); yy = yy(:);
            [xxP, yyP] = meshgrid(xP, yP);
            xxP = xxP(:); yyP = yyP(:);
            FineMesh.xx  = xx;    FineMesh.yy  = yy;
            CoarseMesh.xx = xxP;  CoarseMesh.yy = yyP;
            [~, ~, CoarseMesh.interMatP] = getInterpolationMatrix(xxP, yyP, Nx-2, Ny-2, xx, yy, Nx, Ny);
            FineMesh.interMatU = buildRestrictTruncMat(Nx, Ny);
        end

        %% initial condition -- SAME underlying field at every Nx
        theta0 = localMaterializeTheta0(coeffs, Nx, Ny, Theta0_amplitude);

        Ncoarse = (Nx-2)*(Ny-2); Nfine = Nx*Ny;
        S0.phi   = phi0 * ones(Ncoarse, 1);
        S0.theta = theta0;
        if parameter.useFFT
            [S0.Uf1, S0.Uf2, S0.Up1, S0.Up2, S0.P] = solveVelocityFFT(S0.phi, S0.theta, parameter, D);
        else
            [S0.Uf1, S0.Uf2, S0.Up1, S0.Up2, S0.P] = solveVelocity(S0.phi, S0.theta, parameter, FineMesh, CoarseMesh);
        end


        %% run nSteps_s predictor-corrector steps at fixed dt_s
        S = S0; Sprev = S0;
        flagOk = true;
        tA = tic;
        for s = 1:nSteps_s
            try
                [S, Sprev] = pcStep(S, Sprev, dt_s, parameter, D, FineMesh, CoarseMesh);
            catch ME
                fprintf('  FAILED at Nx=%d step=%d/%d: %s\n', Nx, s, nSteps_s, ME.message);
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
        results(i).Nx = Nx; results(i).Ny = Ny; results(i).kFinal = kFinal;
        results(i).wallTime = wallTime; results(i).flagOk = flagOk;
        wallAll(dtIdx, i) = wallTime;
        fprintf('Nx=Ny=%3d  wall=%.3fs  flagOk=%d\n', Nx, wallTime, flagOk);
    end

    %% ---- analysis for this dt: Uf1 self-convergence vs. the finest Nx ----
    % Error measured against THIS dt's own finest-Nx run (resampled onto
    % that grid), restricted to Uf1 only -- same rationale as
    % NumericsAveragingTheory\spatialConvergenceStudy.m's analysis
    % section, just with extractUf1 isolating one block.
    n = numel(results);
    if ~results(n).flagOk
        warning('spatialConvergenceStudyPC:finestFailed', ...
            'At dt=%.3e, the finest-Nx run (Nx=%d) did not converge -- self-convergence numbers below are meaningless (comparing against a NaN reference).', dt_s, NxList_s(n));
    end
    NxFinest = results(n).Nx; NyFinest = results(n).Ny;
    resampledUf1 = cell(1, n);
    for i = 1:n
        resampledUf1{i} = extractUf1(localResampleState(results(i).kFinal, results(i).Nx, results(i).Ny, NxFinest, NyFinest), NxFinest, NyFinest);
    end
    finalRef = resampledUf1{n};
    refNorm = norm(finalRef);
    fprintf('\n--- Uf1 self-convergence vs finest Nx=Ny=%d at dt=%.3e (expect near-spectral -- see header); ||finestUf1||=%.4e ---\n', NxFinest, dt_s, refNorm);
    for i = 1:n-1
        dAll(dtIdx, i) = norm(resampledUf1{i} - finalRef) / refNorm;
        fprintf('  Nx=Ny=%3d vs finest:  relative ||diff||=%.4e', results(i).Nx, dAll(dtIdx, i));
        if i > 1
            r = results(i).Nx / results(i-1).Nx;   % actual refinement ratio for THIS pair
            fprintf('   ratio=%.3f  observed order p=%.2f', dAll(dtIdx,i-1)/dAll(dtIdx,i), log(dAll(dtIdx,i-1)/dAll(dtIdx,i))/log(r));
        end
        fprintf('\n');
    end
end

%% ===================== plot: one curve per dt =====================
% N on the x-axis and semilogy, not loglog vs h -- spectral (exponential)
% convergence is a straight line on semilog-y against N. See
% NumericsAveragingTheory\spatialConvergenceStudy.m's plot comment for the
% same reasoning.
NxTestedList = NxList_s(1:end-1);
f = figure('Visible', 'off', 'Position', [100 100 800 550]);
markers = {'s-', 'o-', '^-', 'd-', 'v-'};
colors = lines(nDt);
hold on;
for dtIdx = 1:nDt
    semilogy(NxTestedList, dAll(dtIdx,:), markers{mod(dtIdx-1,numel(markers))+1}, ...
        'LineWidth', 1.5, 'MarkerSize', 7, 'Color', colors(dtIdx,:), ...
        'DisplayName', sprintf('\\Deltat=%.1e', dtList_s(dtIdx)));
end
hold off;
% Explicit, not relied-upon-implicitly: semilogy() only sets YScale to
% 'log' when it creates a fresh plot -- with `hold on` active before the
% loop (needed so each dt's curve overlays instead of overwriting), later
% calls just behave like plot() on whatever scale the axes already have.
% Verified this exact gotcha silently breaks log axes elsewhere in this
% project (ModelComparison's cross-model plots) -- setting it explicitly
% here is robust to it regardless of hold state.
set(gca, 'YScale', 'log');
xlabel('N_x'); ylabel('relative ||U^f_1(N_x) - U^f_1(finest N_x)||');
if parameter.useFFT
    solverTag = 'FFT/GMRES';
else
    solverTag = 'dense';
end
title(sprintf('PredictorCorrector (splitting) U^f_1 spatial self-convergence (relative), multiple \\Deltat [%s solvers]', solverTag));
legend('Location', 'northeast'); grid on;

outDir = fullfile(projRoot, 'plotting');
if ~exist(outDir, 'dir'); mkdir(outDir); end
if parameter.useFFT
    outFile = fullfile(outDir, 'pc_spatial_convergence.png');
else
    outFile = fullfile(outDir, 'pc_spatial_convergence_dense.png');
end
saveas(f, outFile);
close(f);
fprintf('\nSaved plot to %s\n', outFile);

%% ===================== wall time =====================
fprintf('\n--- wall time ---\n');
for dtIdx = 1:nDt
    fprintf('  dt=%.3e:\n', dtList_s(dtIdx));
    for i = 1:nNx
        fprintf('    Nx=Ny=%3d:  %.3fs\n', NxList_s(i), wallAll(dtIdx, i));
    end
end

%% ===================== local functions =====================
function u = extractUf1(k, Nx, Ny)
% Isolates the Uf1 block (fluid velocity, x-component) from the full
% 7-block state vector layout [phi;Uf1;Uf2;P;Up1;Up2;theta] shared by
% every solver in this project.
NP = (Nx-2)*(Ny-2); Nf = Nx*Ny;
u = k(NP+1 : NP+Nf);
end

function [S, Sprev] = pcStep(S, Sprev, dt_step, parameter, D, FineMesh, CoarseMesh)
% One predictor-corrector step -- exactly mainPC.m's Step 1-7, inlined
% here so this script can call it without depending on mainPC.m itself
% (a top-level script, not a callable function). Identical to
% temporalConvergenceStudyPC.m's copy of the same local function, except
% it now also dispatches to mainPC.m's dense/direct solver family
% (solveVelocity/solvePhi[CN]/solveTheta[CN]) when parameter.useFFT=0,
% same branching mainPC.m's own time loop does. FineMesh/CoarseMesh are
% unused (and may be passed as []) when useFFT=1.
useFFT = parameter.useFFT;
Uf1_star = S.Uf1; Uf2_star = S.Uf2; Up1_star = S.Up1; Up2_star = S.Up2;

if useFFT
    phi_half = solvePhiFFT(S.phi, Up1_star, Up2_star, dt_step/2, parameter, D);
    theta_half = solveThetaFFT(S.theta, S.phi, Uf1_star, Uf2_star, Up1_star, Up2_star, dt_step/2, parameter, D);
else
    phi_half = solvePhi(S.phi, Up1_star, Up2_star, dt_step/2, parameter, FineMesh, CoarseMesh);
    theta_half = solveTheta(S.theta, S.phi, Uf1_star, Uf2_star, Up1_star, Up2_star, dt_step/2, parameter, CoarseMesh);
end

if useFFT
    x0_h = 1.5*[S.Uf1; S.Uf2; S.Up1; S.Up2; S.P] - 0.5*[Sprev.Uf1; Sprev.Uf2; Sprev.Up1; Sprev.Up2; Sprev.P];
    [Uf1_h, Uf2_h, Up1_h, Up2_h, P_h] = solveVelocityFFT(phi_half, theta_half, parameter, D, x0_h);
else
    [Uf1_h, Uf2_h, Up1_h, Up2_h, P_h] = solveVelocity(phi_half, theta_half, parameter, FineMesh, CoarseMesh);
end

if useFFT
    phi_n1 = solvePhiCNFFT(S.phi, Up1_h, Up2_h, dt_step, parameter, D);
    theta_n1 = solveThetaCNFFT(S.theta, phi_half, Uf1_h, Uf2_h, Up1_h, Up2_h, dt_step, parameter, D);
else
    phi_n1 = solvePhiCN(S.phi, Up1_h, Up2_h, dt_step, parameter, FineMesh, CoarseMesh);
    theta_n1 = solveThetaCN(S.theta, phi_half, Uf1_h, Uf2_h, Up1_h, Up2_h, dt_step, parameter, CoarseMesh);
end

if useFFT
    x0_n1 = 2*[Uf1_h; Uf2_h; Up1_h; Up2_h; P_h] - [S.Uf1; S.Uf2; S.Up1; S.Up2; S.P];
    [Uf1_n1, Uf2_n1, Up1_n1, Up2_n1, P_n1] = solveVelocityFFT(phi_n1, theta_n1, parameter, D, x0_n1);
else
    [Uf1_n1, Uf2_n1, Up1_n1, Up2_n1, P_n1] = solveVelocity(phi_n1, theta_n1, parameter, FineMesh, CoarseMesh);
end

Sprev = S;
S.phi = phi_n1; S.theta = theta_n1;
S.Uf1 = Uf1_n1; S.Uf2 = Uf2_n1; S.Up1 = Up1_n1; S.Up2 = Up2_n1; S.P = P_n1;
end

function coeffs = localBuildConsistentTheta0Seed(cutoff, seed)
% Resolution-independent theta0 "recipe" -- self-contained copy of the
% same logic used in NumericsAveragingTheory\spatialConvergenceStudy.m /
% ModelComparison\buildConsistentTheta0Seed.m (see that file's header for
% the full rationale on why getTheta0.m alone can't be called
% independently at each Nx for a valid spatial convergence study).
refN = 10;
probeN = 256;

rng(seed);
theta_four = (rand(refN, refN) - 0.5) + 0*1i*(rand(refN, refN) - 0.5);
theta_four_shifted = fftshift(theta_four);
centerx = floor(refN/2) + 1;
centery = floor(refN/2) + 1;
for ii = 1:refN
    for jj = 1:refN
        if sqrt((ii-centery)^2 + (jj-centerx)^2) > cutoff
            theta_four_shifted(ii,jj) = 0;
        end
    end
end
base = real(ifft2(ifftshift(theta_four_shifted)));
base = base(:);

coeffs.base = base;
coeffs.refN = refN;
coeffs.cutoff = cutoff;

probe = localSpectralUpsample2D(base, refN, refN, probeN, probeN);
coeffs.normScale = max(abs(probe));
end

function theta = localMaterializeTheta0(coeffs, Nx, Ny, amplitude)
% Build theta0 at a target (Nx,Ny) from the resolution-independent seed --
% self-contained copy of the same logic used elsewhere in this project
% (see localBuildConsistentTheta0Seed's comment above).
if Nx == coeffs.refN && Ny == coeffs.refN
    raw = coeffs.base;
else
    if Nx < coeffs.refN || Ny < coeffs.refN
        error('spatialConvergenceStudyPC:tooCoarse', ...
            'Target grid (%d x %d) is smaller than the seed''s reference resolution (%d x %d) -- localSpectralUpsample2D only upsamples. Lower refN in localBuildConsistentTheta0Seed if you need a coarser target.', ...
            Ny, Nx, coeffs.refN, coeffs.refN);
    end
    raw = localSpectralUpsample2D(coeffs.base, coeffs.refN, coeffs.refN, Ny, Nx);
end

theta = amplitude * raw / coeffs.normScale;
theta = theta(:);
end

function kOut = localResampleState(kIn, Nx1, Ny1, Nx2, Ny2)
% Spectrally resample a full 7-block state vector (layout used by
% mainPC.m: [phi;Uf1;Uf2;P;Up1;Up2;theta], with P reduced by the mean-
% value gauge to Ncoarse-1 entries) from an (Nx1,Ny1) grid to an (Nx2,Ny2)
% grid -- self-contained copy of ModelComparison\resampleState.m's logic
% (same state layout as NumericsAveragingTheory's, which is why that
% file's copy works for both models unmodified). Output holds phi/P in
% FULL (unreduced) form, since this is only ever used for comparison.
% Also works as a pure "expand P to full form, no resolution change" call
% when Nx1==Nx2, Ny1==Ny2.
NP1 = (Nx1-2)*(Ny1-2); Np1 = NP1-1; Nf1 = Nx1*Ny1;

phi1    = kIn(1:NP1);
Uf1_1   = kIn(NP1+1 : NP1+Nf1);
Uf2_1   = kIn(NP1+Nf1+1 : NP1+2*Nf1);
Pred1   = kIn(NP1+2*Nf1+1 : NP1+2*Nf1+Np1);
Up1_1   = kIn(NP1+2*Nf1+Np1+1 : NP1+2*Nf1+Np1+Nf1);
Up2_1   = kIn(NP1+2*Nf1+Np1+Nf1+1 : NP1+2*Nf1+Np1+2*Nf1);
theta1  = kIn(NP1+2*Nf1+Np1+2*Nf1+1 : NP1+2*Nf1+Np1+3*Nf1);

Pfull1 = zeros(NP1,1);
Pfull1(1:Np1) = Pred1;
Pfull1(NP1) = -sum(Pred1);

phi2   = localSpectralUpsample2D(phi1,   Ny1-2, Nx1-2, Ny2-2, Nx2-2);
Pfull2 = localSpectralUpsample2D(Pfull1, Ny1-2, Nx1-2, Ny2-2, Nx2-2);
Uf1_2  = localSpectralUpsample2D(Uf1_1,  Ny1,   Nx1,   Ny2,   Nx2);
Uf2_2  = localSpectralUpsample2D(Uf2_1,  Ny1,   Nx1,   Ny2,   Nx2);
Up1_2  = localSpectralUpsample2D(Up1_1,  Ny1,   Nx1,   Ny2,   Nx2);
Up2_2  = localSpectralUpsample2D(Up2_1,  Ny1,   Nx1,   Ny2,   Nx2);
theta2 = localSpectralUpsample2D(theta1, Ny1,   Nx1,   Ny2,   Nx2);

kOut = [phi2; Uf1_2; Uf2_2; Pfull2; Up1_2; Up2_2; theta2];
end

function xf = localSpectralUpsample2D(xc, Nyc, Nxc, Nyf, Nxf)
% Coarse Nyc x Nxc -> fine Nyf x Nxf spectral lift, real space in and out.
% Ax=1 throughout (this codebase's fixed convention -- see
% fourierGrid(1,N) everywhere). Self-contained copy of
% PredictorCorrector\common\spectralUpsample2D-equivalent logic (same as
% ModelComparison\spectralUpsample2D.m); deliberately named "local..."
% (not a separate file) to avoid colliding with
% PredictorCorrector\common\fourierUpsample2D.m, a different function
% used internally elsewhere in this codebase.
Ax = 1; L = 2*Ax;
Xc = fft2(reshape(xc, Nyc, Nxc));

if Nyc == Nyf
    tmp = Xc;
else
    half1 = Nyc/2;
    shift1 = 2*Ax*(1/Nyf - 1/Nyc);
    tmp = zeros(Nyf, Nxc);
    k_pos1 = (0:half1-1)';
    tmp(1:half1,:) = Xc(1:half1,:) .* exp(1i*2*pi*k_pos1*shift1/L);
    phase_nyq1 = exp(1i*2*pi*half1*shift1/L);
    tmp(half1+1,:)         = Xc(half1+1,:)/2 * phase_nyq1;
    tmp(Nyf-half1+1,:)     = tmp(Nyf-half1+1,:) + Xc(half1+1,:)/2 * conj(phase_nyq1);
    k_neg1 = (-(half1-1):-1)';
    tmp(Nyf-half1+2:Nyf,:) = Xc(half1+2:Nyc,:) .* exp(1i*2*pi*k_neg1*shift1/L);
    tmp = tmp * (Nyf/Nyc);
end

tmpT = tmp.';
if Nxc == Nxf
    XfT = tmpT;
else
    half2 = Nxc/2;
    shift2 = 2*Ax*(1/Nxf - 1/Nxc);
    XfT = zeros(Nxf, Nyf);
    k_pos2 = (0:half2-1)';
    XfT(1:half2,:) = tmpT(1:half2,:) .* exp(1i*2*pi*k_pos2*shift2/L);
    phase_nyq2 = exp(1i*2*pi*half2*shift2/L);
    XfT(half2+1,:)         = tmpT(half2+1,:)/2 * phase_nyq2;
    XfT(Nxf-half2+1,:)     = XfT(Nxf-half2+1,:) + tmpT(half2+1,:)/2 * conj(phase_nyq2);
    k_neg2 = (-(half2-1):-1)';
    XfT(Nxf-half2+2:Nxf,:) = tmpT(half2+2:Nxc,:) .* exp(1i*2*pi*k_neg2*shift2/L);
    XfT = XfT * (Nxf/Nxc);
end
Xf = XfT.';

xf = real(ifft2(Xf));
xf = xf(:);
end
