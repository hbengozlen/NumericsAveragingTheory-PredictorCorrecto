function [Uf1, Uf2, Up1, Up2, P] = solveVelocityFFT(phi, theta, parameter, D, x0)
% Matrix-free, FFT-based twin of solveVelocity.m -- same equations, same
% signs, same boolactive branch, same three stability fixes (velocity gauge
% pin, Nyquist dealiasing, row scaling); solveVelocity.m itself is untouched.
%
% solveVelocity.m builds dense DXX=DX*DX etc. and a big dense A_sys, then
% solves it with backslash. Here, DX*x/DY*x become FFT-based spectral
% differentiation and interMatP*x/interMat*x become spectral lift/restrict
% (fourierDiffOps.m), and the whole dense matrix is replaced by a function
% Av(v) that applies the operator's action on a candidate vector -- so
% A_sys is never assembled. The linear system is then solved with GMRES,
% preconditioned by the same operator with phi frozen to its mean
% (buildVelocityPreconditioner.m: constant-coefficient -> exactly
% block-diagonal in Fourier space, one small 5x5 solve per shared
% wavenumber instead of one huge dense factorization).
%
% Split across files for readability/reuse:
%   fourierDiffOps.m               -- spectral derivative/lift/restrict
%                                      ops, reusable by any solver on this
%                                      grid.
%   buildVelocityPreconditioner.m  -- the frozen-phibar 5x5-per-mode Minv.
%   computeVelocityRHS.m           -- the explicit right-hand side.
%   velocityAv.m                   -- the true, variable-coefficient
%                                      matrix-free operator.
%   velocityPinv.m                 -- applies the preconditioner.
%   solveVelocityFFT.m (here)      -- orchestration only.
%
% D = fourierDiffOps(Nx,Ny), computed ONCE by the caller (e.g. mainPC.m, at
% setup, alongside parameter.DX/DY/D2X/D2Y) and passed in here -- Nx,Ny are
% fixed for an entire run, so recomputing D on every call (this function is
% called twice per timestep) would just repeat identical, deterministic
% work. fourierDiffOps.m itself has no internal cache; the caller owns D.
%
% x0 (optional): warm-start initial guess for GMRES, in PHYSICAL units
% ([Uf1;Uf2;Up1;Up2;P], same layout as sol below) -- e.g. mainPC.m passes
% the previous timestep's converged solution, since consecutive timesteps
% change little. Under RIGHT preconditioning (the default), GMRES solves
% for y where x=velocityPinv(y), so x0 is first mapped through
% velocityPfwd.m (M's forward, uninverted block) into y0=velocityPfwd(x0)
% before being handed to gmres -- passing x0 straight through would target
% the wrong variable. Under LEFT preconditioning, GMRES solves for the
% physical vector directly, so x0 is passed through unchanged. Omit or
% pass [] to start from zero (MATLAB's gmres default).
%
A_param    = parameter.A;
etaf       = parameter.etaf;
etab       = parameter.etab;
R          = parameter.R;
b          = parameter.b;
V0         = parameter.V0;
Db         = parameter.Db;
PrefPt     = parameter.PrefPt;
Nx         = parameter.Nx;
Ny         = parameter.Ny;
boolactive = parameter.boolactive;
boolpusher = parameter.boolpusher;
alpha      = -2*boolpusher + 1;

Nfine = Nx*Ny;
NP    = (Nx-2)*(Ny-2);
Np    = NP - 1;

%% Profiling: setup (everything before gmres starts) vs. time spent inside
% Av/Pinv once gmres is actually iterating. Accumulated in perfCounter's
% own persistent state (named 'solveVelocityFFT') across every call --
% no global variable -- printed once at the end of the simulation by
% mainPC.m via perfCounter(...,'get'), not per-call.
tFuncStart = tic;

%% Pressure gauge: mean(P)=PrefPt, via the cheap zero-mean basis v_i=e_i-e_NP
% (i=1..Np) instead of null(ones(1,NP)) -- same zero-mean subspace (every
% embedded vector still sums to 0 exactly), but embed/extract become plain
% index ops instead of an NPxNp dense matrix-vector product:
%   embed:   dPfull = [dP; -sum(dP)]                  (was NmatP*dP)
%   extract: out    = v(1:Np) - v(NP)                 (was NmatP'*v)
% Verified this preserves both properties the DC-mode fix at this pressure
% row/column relies on: sum(dPfull)=0 exactly (so R5's DC coefficient into
% Pinv is always 0), and the extract step is invariant to any constant
% added to v (so it's blind to whatever pinv's still-rank-deficient row 5
% returns for pressure's mean at the 4 corner modes) -- no NmatP-specific
% math is used, just the two identities above, so no matrix is ever formed.
Ptilde = PrefPt * ones(NP,1);   % mean(Ptilde) = PrefPt exactly

%% Interpolated quantities on fine grid
phiext = D.liftP(phi);
phibar = mean(phi);              % frozen value used only by the preconditioner
d1 = cos(theta);
d2 = sin(theta);

ksi_A2 = 2*A_param^2/R^2 * phiext;
ksi_Ab = 2*A_param*b/R^2 * phiext;

%% Row-scaling estimate (same role as in solveVelocity.m)
Mscale = max(abs(ksi_A2));
kMax   = max(abs(D.KX(:)));              % largest single first-derivative eigenvalue
rowScale = Mscale / kMax;                % cheap proxy for norm(momentum)/norm(continuity)

RefPtIDXfine = Ny*(Nx/2 - 1) + Ny/2;

rhs = computeVelocityRHS(phiext, ksi_A2, ksi_Ab, d1, d2, alpha, boolactive, ...
    A_param, R, V0, Db, Ptilde, Np, RefPtIDXfine, D);

[Minv, Mblock] = buildVelocityPreconditioner(phibar, parameter, D);

%% Bundle everything velocityAv/velocityPinv/velocityPfwd need -- each
% takes this ctx explicitly instead of capturing it via closure, so their
% signatures are self-documenting and they're callable/testable in
% isolation. M is only ever read by velocityPfwd.m (warm-start), not by
% velocityAv/velocityPinv's own hot loop.
ctx = struct('Nx', Nx, 'Ny', Ny, 'Nfine', Nfine, 'NP', NP, 'Np', Np, ...
    'D', D, 'ksi_A2', ksi_A2, 'phiext', phiext, 'rowScale', rowScale, ...
    'RefPtIDXfine', RefPtIDXfine, 'etab', etab, 'etaf', etaf, ...
    'Minv', Minv, 'M', Mblock, 'KXY', D.KX .* D.KY);

perfCounter('solveVelocityFFT', 'add', 'setup', toc(tFuncStart));
perfCounter('solveVelocityFFT', 'add', 'calls', 1);

%% Solve, matrix-free. useRightPrecond selects which of the two GMRES
% preconditioning modes below runs -- MATLAB's gmres only exposes LEFT
% preconditioner factors via M1/M2 (combined into one left factor
% M=M1*M2); there is no separate right slot, so RIGHT preconditioning is
% instead built by hand: solve (Av*Pinv)*y=rhs for y, then recover
% x=Pinv*y. Right preconditioning's key advantage: the residual GMRES
% minimizes, Av*(Pinv*y)-rhs = Av*x-rhs, IS the TRUE residual of the
% original system, not a preconditioned stand-in -- so gmresRelres is
% directly trustworthy under it. LEFT preconditioning's relres is instead
% the PRECONDITIONED residual, verified to under-report the true one by
% orders of magnitude and unevenly across equations (x-momentum rows ~10%
% true error at a preconditioned relres of 1e-9) -- kept here only for
% comparison/regression-testing that failure mode, not for general use.
% Set via parameter.useRightPrecond (mainPC.m), not hardcoded here.
useRightPrecond = parameter.useRightPrecond;
haveX0 = nargin >= 5 && ~isempty(x0);

n = 4*Nfine + Np;
tol = parameter.gmresTol;      % single shared value, set once in mainPC.m -- see its header comment
maxit = parameter.gmresMaxIt;  % single shared value, set once in mainPC.m -- see its header comment
% restart=parameter.gmresRestartVelocity (100, not 30): at Nx=Ny=40-100 this
% was measured to matter a LOT -- 30 forces GMRES to discard its whole
% Krylov subspace on every restart, costing 10-15x more total matrix-vector
% products (or outright failing to converge within maxit at Nx=Ny=100,
% flag=1) versus 100, which stays comfortably inside the range that
% converges cleanly (flag=0) without the borderline stagnation warnings
% seen at 150+. See mainPC.m's header comment for why this is its own field,
% not shared with solvePhiFFT/solveThetaFFT's restart.
restart = min(parameter.gmresRestartVelocity, n);
Av_phys = @(v) velocityAv(v, ctx);
Pinv    = @(r) velocityPinv(r, ctx);
AvPinv  = @(y) velocityAv(velocityPinv(y, ctx), ctx);   % right-preconditioned operator

% Iterative refinement (defect correction) instead of demanding
% parameter.gmresTol directly from a single GMRES call. In the fully-
% coupled Newton solver, GMRES sits INSIDE an inexact-Newton iteration, so
% a loose inner linear tolerance is fine -- the next Newton step mops up
% whatever residual it leaves behind. This solve has no such outer loop:
% it's a one-shot linear solve feeding straight into the predictor/
% corrector stage, so historically this asked GMRES itself for the full
% target (e.g. 1e-9) in one call -- landing squarely in the ~1e-7 zone
% where MATLAB's stagnation heuristic false-alarms regardless of the
% requested tol (it's a step-size check, not a distance-to-target check;
% verified in tests/diagnose_gmres_flags.m). Instead: every individual
% gmres() call below only ever targets looseTol, comfortably above that
% false-stagnation zone, so it always returns cleanly; the TRUE residual
% (via the exact operator velocityAv -- not gmres's own relres, which
% under LEFT preconditioning is the preconditioned residual, not the true
% one, per this file's header comment) is measured directly afterward and,
% if still short of the real target, GMRES solves again for just the
% leftover correction. This borrows the same loose-inner/tight-outer
% structure inexact-Newton-GMRES gets for free from its Newton loop, built
% explicitly here since there's no Newton loop to borrow it from.
requestedTol = tol;                  % parameter.gmresTol, the real target
looseTol = max(1e-6, requestedTol);  % floor: never ask any single gmres call for anything
                                      % stricter than 1e-6 (one order of magnitude looser than
                                      % the empirically-observed ~1e-7 false-stagnation zone,
                                      % tests/diagnose_gmres_flags.m) -- EXCEPT when requestedTol
                                      % itself is already looser than that (e.g. a study
                                      % deliberately running at gmresTol=1e-3), in which case
                                      % there's no reason to demand more precision per call than
                                      % the real target asks for. max(), not min(): this must
                                      % never resolve to something TIGHTER than 1e-6 -- notably
                                      % including when requestedTol is TIGHTER than 1e-6 (e.g.
                                      % the convergence studies' 1e-13), which is the common case
                                      % this floor exists to protect in the first place.
maxRounds = 10;
roundCount = 0;
rhsNorm = norm(rhs);

% Suppress gmres's own "stopped without converging"/stagnation console
% warning -- flag/relres/roundCount are already reported ourselves below.
% Scoped to just this solve and restored via onCleanup even on error.
warnState = warning('off', 'all');
cleanupWarn = onCleanup(@() warning(warnState)); %#ok<NASGU>

% Every round -- including the first -- solves for a CORRECTION dx against
% the CURRENT true residual `res`, rather than solving the full system
% fresh each time: starting `sol` at x0 (warm start) folds x0's benefit
% straight into round 1's res (no separate velocityPfwd-mapped y0 needed),
% and each round's target tol is capped by requestedTol/trueRelres -- the
% loosest tol that still reaches requestedTol overall from here, given
% this round hits its own target exactly. This matters because reusing a
% FIXED looseTol every round (the previous version of this loop) compounds
% wrongly: once round 1 already gets true relres down to ~looseTol, asking
% a second round for ANOTHER full looseTol reduction relative to that
% already-small residual demands far more decimal digits of accuracy than
% actually needed to clear requestedTol, costing many more restart cycles
% for no benefit -- exactly the "refinement needs a lot of restarts"
% symptom this fixes. max(...) still never drops below looseTol, so no
% round is ever pushed back into the false-stagnation zone either.
if haveX0
    sol = x0;
else
    sol = zeros(n,1);
end
res = rhs - Av_phys(sol);
trueRelres = norm(res) / rhsNorm;
gmresFlag = 0; gmresRelres = trueRelres; gmresIter = [0 0];

while trueRelres > requestedTol && roundCount < maxRounds
    roundCount = roundCount + 1;
    roundTol = max(looseTol, requestedTol / trueRelres);
    if useRightPrecond
        [dy, gmresFlag, gmresRelres, gmresIter] = gmres(AvPinv, res, restart, roundTol, maxit);
        dx = velocityPinv(dy, ctx);
    else
        [dx, gmresFlag, gmresRelres, gmresIter] = gmres(Av_phys, res, restart, roundTol, maxit, [], Pinv);
    end
    sol = sol + dx;
    res = rhs - Av_phys(sol);
    trueRelres = norm(res) / rhsNorm;
end

fprintf('  solveVelocityFFT: gmres flag=%d relres=%.2e true=%.2e iters=%d/%d rounds=%d/%d\n', ...
    gmresFlag, gmresRelres, trueRelres, gmresIter(1), gmresIter(2), roundCount, maxRounds);

Uf1 = sol(1           : Nfine);
Uf2 = sol(Nfine+1     : 2*Nfine);
Up1 = sol(2*Nfine+1   : 3*Nfine);
Up2 = sol(3*Nfine+1   : 4*Nfine);
P   = sol(4*Nfine+1   : 4*Nfine+Np);
end
