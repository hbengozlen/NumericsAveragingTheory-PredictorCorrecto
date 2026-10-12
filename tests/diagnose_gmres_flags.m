%% diagnose_gmres_flags.m
% Confirms/refutes the hypothesis that temporalConvergenceStudyPC.m's /
% spatialConvergenceStudyPC.m's ~1e-8 relative-error plateau is caused by
% GMRES's flag=3 false-stagnation exit firing well before the requested
% gmresTol=1e-14, across the SAME pcStep sequence those studies actually
% run. Captures every "gmres flag=%d relres=%.2e" line the 5 solveXxxFFT
% solvers already print (evalc, no new instrumentation needed in
% production code), tabulates the flag distribution, and reports the
% relres achieved specifically by flag=3 calls -- if those cluster around
% ~1e-8-1e-9 while flag=0 calls (if any occur at all) reach much tighter
% values, that confirms flag=3 is the dominant, artificial floor rather
% than a genuine achievable-precision ceiling.

clear; clc;
projRoot = 'Q:\Code Averaging Theory\Averaging Theroy Numerics\PredictorCorrector';
addpath(genpath(projRoot));

%% Same config as temporalConvergenceStudyPC.m's current setup
parameter.A=200e-6; parameter.etaf=1e-3; parameter.etab=1e-3;
parameter.R=0.7e-6; parameter.L=4e-6; parameter.b=4e-6; parameter.V0=5e-6;
parameter.ksif=-1; parameter.elasPara=36e-12; parameter.Db=1e-7*1e-4*20;
parameter.PrefPt=100; parameter.boolactive=1; parameter.boolpusher=0;
parameter.useRightPrecond=1;
parameter.gmresMaxIt = 100;
parameter.gmresRestartVelocity = 100;
parameter.gmresRestartScalar   = 30;
parameter.gmresTol=1e-14;   % same as temporalConvergenceStudyPC.m's current config

phi0 = 0.15;
Nx = 100; Ny = 100;   % same grid as temporalConvergenceStudyPC.m's current config
Theta0_cutoff = 3; Theta0_amplitude = 1;
seed = 20260806;

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

rng(seed);
theta0 = getTheta0(Nx, Ny, Theta0_cutoff, Theta0_amplitude);
theta0 = theta0(:);

Ncoarse = (Nx-2)*(Ny-2); Nfine = Nx*Ny;
S.phi = phi0*ones(Ncoarse,1);
S.Uf1 = zeros(Nfine,1); S.Uf2 = zeros(Nfine,1);
S.P = parameter.PrefPt*ones(Ncoarse-1,1);
S.Up1 = zeros(Nfine,1); S.Up2 = zeros(Nfine,1);
S.theta = theta0;
Sprev = S;

nStepsDiag = 8;   % enough steps to see whether the flag=3/relres pattern
                  % is a persistent feature of the whole run, not just a
                  % first-step transient (see the earlier "does an
                  % under-converged first step contaminate later ones"
                  % discussion -- this checks the OPPOSITE direction: is
                  % it happening throughout, not just at step 1).
dt = 1e-5/8;      % matches nSub=8 in temporalConvergenceStudyPC.m's own nSubList

allFlags = []; allRelres = []; allWhich = {};
for s = 1:nStepsDiag
    txt = evalc('[S, Sprev] = pcStep(S, Sprev, dt, parameter, D);');
    fprintf('--- step %d ---\n%s', s, txt);
    lines = strsplit(txt, newline);
    for L = 1:numel(lines)
        m = regexp(lines{L}, '^\s*(\w+):\s+gmres flag=(\d+) relres=([\d.eE+-]+)', 'tokens', 'once');
        if ~isempty(m)
            allWhich{end+1}  = m{1};              %#ok<AGROW>
            allFlags(end+1)  = str2double(m{2});   %#ok<AGROW>
            allRelres(end+1) = str2double(m{3});   %#ok<AGROW>
        end
    end
end

fprintf('\n=== Flag distribution across %d GMRES calls (%d PC steps x 5 solvers) ===\n', numel(allFlags), nStepsDiag);
for f = unique(allFlags)
    mask = allFlags == f;
    fprintf('  flag=%d: %3d/%3d calls, relres range [%.3e, %.3e], median %.3e\n', ...
        f, sum(mask), numel(allFlags), min(allRelres(mask)), max(allRelres(mask)), median(allRelres(mask)));
end
fprintf('\nRequested gmresTol = %.3e\n', parameter.gmresTol);

fprintf('\n--- breakdown by solver ---\n');
uniqueWhich = unique(allWhich);
for k = 1:numel(uniqueWhich)
    w = uniqueWhich{k};
    mask = strcmp(allWhich, w);
    f3 = allFlags(mask) == 3;
    fprintf('  %-18s  %2d calls, %2d with flag=3 (relres median %.3e when flag=3)\n', ...
        w, sum(mask), sum(f3), median(allRelres(mask & (allFlags==3))));
end

%% ===================== local function =====================
function [S, Sprev] = pcStep(S, Sprev, dt_step, parameter, D)
% Identical copy of temporalConvergenceStudyPC.m's/spatialConvergenceStudyPC.m's
% local pcStep -- exactly mainPC.m's Step 1-7.
Uf1_star = S.Uf1; Uf2_star = S.Uf2; Up1_star = S.Up1; Up2_star = S.Up2;

phi_half = solvePhiFFT(S.phi, Up1_star, Up2_star, dt_step/2, parameter, D);
theta_half = solveThetaFFT(S.theta, S.phi, Uf1_star, Uf2_star, Up1_star, Up2_star, dt_step/2, parameter, D);

x0_h = 1.5*[S.Uf1; S.Uf2; S.Up1; S.Up2; S.P] - 0.5*[Sprev.Uf1; Sprev.Uf2; Sprev.Up1; Sprev.Up2; Sprev.P];
[Uf1_h, Uf2_h, Up1_h, Up2_h, P_h] = solveVelocityFFT(phi_half, theta_half, parameter, D, x0_h);

phi_n1 = solvePhiCNFFT(S.phi, Up1_h, Up2_h, dt_step, parameter, D);
theta_n1 = solveThetaCNFFT(S.theta, phi_half, Uf1_h, Uf2_h, Up1_h, Up2_h, dt_step, parameter, D);

x0_n1 = 2*[Uf1_h; Uf2_h; Up1_h; Up2_h; P_h] - [S.Uf1; S.Uf2; S.Up1; S.Up2; S.P];
[Uf1_n1, Uf2_n1, Up1_n1, Up2_n1, P_n1] = solveVelocityFFT(phi_n1, theta_n1, parameter, D, x0_n1);

Sprev = S;
S.phi = phi_n1; S.theta = theta_n1;
S.Uf1 = Uf1_n1; S.Uf2 = Uf2_n1; S.Up1 = Up1_n1; S.Up2 = Up2_n1; S.P = P_n1;
end
