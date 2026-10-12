%% test_condJ_approx.m
% Checks how good solveVelocityFFT.m's cheap condJ proxy actually is,
% against the TRUE condition number of the operator it's meant to estimate.
%
% solveVelocityFFT.m computes:
%   delta = max(abs(phi - phibar)) / phibar;
%   condJ = (1+delta) / (1-delta);
% as a free (no dense matrix, no factorization) stand-in for "how
% ill-conditioned is the RIGHT-preconditioned system Av*Pinv (the operator
% GMRES actually iterates on)". It's explicitly documented there as a
% heuristic ("same 'ratio of extremes' shape as the reference codebase's
% own condJ shortcut"), not a derived bound -- this test builds the actual
% dense Av*Pinv operator (by probing velocityAv/velocityPinv with unit
% vectors, same style as test_velocity_fft_accuracy.m /
% test_preconditioner_rank_deficiency.m) and calls MATLAB's cond() on it
% directly, across a sweep of phi's spatial spread (delta), to see whether
% condJ tracks the true value -- in trend, in order of magnitude, or
% neither.
%
% Also reports cond(M) (the frozen preconditioner block alone, dense) at
% each delta as a sanity check: M is built from phibar only, so cond(M)
% should stay roughly CONSTANT across the whole sweep regardless of delta
% -- if it doesn't, something in this test's setup (not condJ itself) is
% wrong. The growth (if any) in cond(Av*Pinv) as delta grows is entirely
% the mismatch between the TRUE variable-coefficient operator and the
% FROZEN-phibar preconditioner -- exactly what condJ is trying to predict.

clear; clc;
addpath('../velocity', '../common');   % velocityAv.m, velocityPinv.m, buildVelocityPreconditioner.m, fourierDiffOps.m, etc.

%% Test case setup (small grid -- dense n x n probing, so keep Nx,Ny small)
parameter.A    = 200e-6;
parameter.etaf = 1e-3;
parameter.etab = 1e-4;
parameter.R    = 0.7e-6;

Nx = 10; Ny = 10;
parameter.Nx = Nx; parameter.Ny = Ny;
D = fourierDiffOps(Nx, Ny);

Nfine = Nx*Ny; NP = (Nx-2)*(Ny-2); Np = NP - 1;
n = 4*Nfine + Np;

phibar = 0.15;   % fixed mean -- only phi's SPREAD around it (delta) varies below

% Smooth spatial shape over the coarse (Ny-2) x (Nx-2) grid, normalized to
% max(abs(shape))=1 -- a real phi field's spatial structure, not noise, so
% max(abs(phi-phibar))/phibar comes out to EXACTLY the target delta below.
[Jc, Ic] = meshgrid(0:Nx-3, 0:Ny-3);
shape = cos(2*pi*Ic/(Ny-2)) .* cos(2*pi*Jc/(Nx-2));
shape = shape / max(abs(shape(:)));

deltaList = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95];

fprintf('=== condJ proxy vs. TRUE cond(), dense n=%d, Nx=Ny=%d ===\n\n', n, Nx);
fprintf('%-8s %-14s %-14s %-14s %-14s %-10s\n', 'delta', 'condJ (proxy)', 'cond(Av) alone', 'cond(AvPinv)', 'cond(M) alone', 'precond factor');

results = nan(numel(deltaList), 5);
I_n = eye(n);
for k = 1:numel(deltaList)
    delta = deltaList(k);
    phi = phibar * (1 + delta*shape(:));

    condJ = (1+delta) / (1-delta);   % solveVelocityFFT.m's own formula, delta<1 guaranteed by construction here

    ctx = buildCtxForTest(phi, parameter, D);

    Av_dense = zeros(n, n);
    AvPinv_dense = zeros(n, n);
    for j = 1:n
        Av_dense(:,j)     = velocityAv(I_n(:,j), ctx);
        AvPinv_dense(:,j) = velocityAv(velocityPinv(I_n(:,j), ctx), ctx);
    end
    condAv    = cond(Av_dense);     % the UNPRECONDITIONED operator solveVelocityFFT.m would face without Pinv
    trueCond  = cond(AvPinv_dense); % what GMRES actually iterates on (right-preconditioned)

    condM = maxCondOverModes(ctx.M);     % worst-conditioned single 5x5 mode block, dense cond() each

    results(k,:) = [delta, condJ, condAv, trueCond, condM];
    fprintf('%-8.3g %-14.4e %-14.4e %-14.4e %-14.4e %-10.3f\n', delta, condJ, condAv, trueCond, condM, condAv/trueCond);
end

fprintf(['\nIf true cond(AvPinv) tracks condJ in ORDER OF MAGNITUDE and both grow with\n' ...
         'delta at a similar rate, condJ is a reasonable (if not tight) proxy for\n' ...
         'deciding when the preconditioner is starting to mismatch. If cond(M) alone\n' ...
         'stays flat across the sweep (as it should -- M only depends on phibar) while\n' ...
         'true cond(AvPinv) grows much faster or slower than condJ predicts, condJ is\n' ...
         'not capturing the actual mismatch mechanism well.\n']);

%% ===================== part 2: does condJ actually track GMRES iterations? =====================
% cond() (ratio of extreme SINGULAR VALUES) is the right diagnostic for a
% NORMAL matrix's GMRES convergence rate; Av*Pinv here is a general
% non-normal, INDEFINITE saddle-point operator, for which classical cond()
% is known to be a poor/irrelevant predictor of GMRES's actual iteration
% count (GMRES convergence for non-normal operators is governed by the
% spectrum's clustering / pseudospectra, not sigma_max/sigma_min). So part
% 1's huge, condJ-uncorrelated cond(AvPinv) values don't necessarily mean
% condJ is a bad proxy for what it's actually used for in
% solveVelocityFFT.m: a quick heads-up alongside the REAL thing that
% matters in practice -- how many GMRES iterations does this delta
% actually cost. That's what this part checks directly.
fprintf('\n=== Part 2: condJ vs. ACTUAL GMRES iterations (same delta sweep) ===\n\n');
fprintf('%-8s %-14s %-8s %-10s %-14s\n', 'delta', 'condJ (proxy)', 'flag', 'iters', 'relres');

rng(42);
rhs = randn(n, 1);
restart = min(100, n);
tol = 1e-10;
maxit = 100;

iterResults = nan(numel(deltaList), 2);   % [totalIters, condJ]
for k = 1:numel(deltaList)
    delta = deltaList(k);
    phi = phibar * (1 + delta*shape(:));
    condJ = (1+delta) / (1-delta);

    ctx = buildCtxForTest(phi, parameter, D);
    AvPinv = @(y) velocityAv(velocityPinv(y, ctx), ctx);
    [~, flag, relres, iter] = gmres(AvPinv, rhs, restart, tol, maxit);
    totalIters = (iter(1)-1)*restart + iter(2);

    iterResults(k,:) = [totalIters, condJ];
    fprintf('%-8.3g %-14.4e %-8d %-10d %-14.4e\n', delta, condJ, flag, totalIters, relres);
end

rho = corr(iterResults(:,2), iterResults(:,1), 'Type', 'Spearman');
fprintf('\nSpearman rank correlation between condJ and actual GMRES iteration count: %.3f\n', rho);
fprintf(['(1.0 = condJ ranks deltas in EXACTLY the same order as GMRES iteration count\n' ...
         'actually does -- i.e. condJ is a trustworthy RELATIVE indicator of which\n' ...
         'deltas will be harder, even though Part 1 shows it is NOT a quantitatively\n' ...
         'accurate cond() estimate. Near 0 or negative would mean condJ is not even a\n' ...
         'reliable ORDERING signal for this operator.)\n']);

%% ===================== part 3: does scaling the gauge/mask rows help? =====================
% Row 5 (continuity) already gets a deliberate rowScale = Mscale/kMax
% correction (velocityAv.m/buildVelocityPreconditioner.m's own comments:
% "row-scaling estimate ... same role as in solveVelocity.m") specifically
% to keep it comparable in magnitude to the momentum rows. The velocity
% GAUGE PIN (out1/out2(RefPtIDXfine) = mean(dUf1/dUf2), and M's matching
% M(1,1,1,1)=1 / M(2,2,1,1)=1 decoupling) and the PRESSURE MASK's decoupled
% diagonal (M(5,5,...)=1 at the DC mode and the whole gap band) never got
% the same treatment -- they're hardcoded to unscaled 1's/means, sitting
% next to rows whose entries scale like kappaPhibar or KX2 (Part 1/2's
% likely explanation for cond(Av) staying ~2-4e6 almost independent of
% delta). This section builds SCALED variants (gauge and mask rows
% multiplied by the same Mscale used for rowScale, since Part 1 showed
% Mscale is the right ballpark for "typical row magnitude" in this system)
% and reruns the identical delta sweep, local-only -- solveVelocityFFT.m /
% buildVelocityPreconditioner.m / velocityAv.m themselves are untouched.
fprintf('\n=== Part 3: scaling the gauge-pin and mask rows -- does it help? ===\n\n');
fprintf('%-8s %-14s %-14s %-10s %-6s %-10s %-14s\n', 'delta', 'cond(Av) scaled', 'cond(AvPinv) scaled', 'iters(unscl)', 'flag', 'iters(scl)', 'relres(scl)');

scaledResults = nan(numel(deltaList), 4);
for k = 1:numel(deltaList)
    delta = deltaList(k);
    phi = phibar * (1 + delta*shape(:));

    ctx = buildCtxForTest(phi, parameter, D);
    ctx.gaugeScale = ctx.rowScale * max(abs(D.KX(:)));   % = Mscale, recovered from rowScale*kMax (Part 1's "typical row magnitude")

    [MinvS, MblockS] = buildVelocityPreconditionerScaled(phibar, parameter, D, ctx.gaugeScale);
    ctxS = ctx; ctxS.Minv = MinvS; ctxS.M = MblockS;

    Av_dense = zeros(n, n); AvPinv_dense = zeros(n, n);
    for j = 1:n
        Av_dense(:,j)     = velocityAvScaled(I_n(:,j), ctxS);
        AvPinv_dense(:,j) = velocityAvScaled(velocityPinv(I_n(:,j), ctxS), ctxS);
    end
    condAvS   = cond(Av_dense);
    trueCondS = cond(AvPinv_dense);

    AvPinvFun = @(y) velocityAvScaled(velocityPinv(y, ctxS), ctxS);
    [~, flagS, relresS, iterS] = gmres(AvPinvFun, rhs, restart, tol, maxit);
    totalItersS = (iterS(1)-1)*restart + iterS(2);

    scaledResults(k,:) = [delta, condAvS, trueCondS, totalItersS];
    fprintf('%-8.3g %-14.4e %-14.4e %-10d %-6d %-10d %-14.4e\n', delta, condAvS, trueCondS, iterResults(k,1), flagS, totalItersS, relresS);
end

fprintf(['\nCompare this cond(Av)/cond(AvPinv) column to Part 1''s ~2-4e6: if scaling the\n' ...
         'gauge-pin and mask rows to match the physics rows'' magnitude drops cond() by\n' ...
         'orders of magnitude, that confirms those unscaled rows (not any genuine\n' ...
         'phi-driven coupling difficulty) were the dominant artifact. Compare the two\n' ...
         'iteration columns to see whether this is just a cosmetic cond() fix or an\n' ...
         'actual practical improvement in GMRES''s real iteration count.\n']);

%% ===================== local functions (part 3) =====================
function [Minv, M] = buildVelocityPreconditionerScaled(phibar, parameter, D, scaleVal)
% Identical to buildVelocityPreconditioner.m, except the DC-mode velocity
% gauge pin and every pressure-decoupled diagonal entry are set to
% scaleVal instead of a bare 1 -- see Part 3's header comment above.
Nx = parameter.Nx; Ny = parameter.Ny;
A_param = parameter.A; R = parameter.R;
etab = parameter.etab; etaf = parameter.etaf;

KX = D.KX; KY = D.KY; KX2 = D.KX2; KY2 = D.KY2;
kappaPhibar = 2*A_param^2/R^2*phibar;

M = zeros(5,5,Ny,Nx);
M(1,1,:,:) = -kappaPhibar + 2*KX2+KY2;   M(1,2,:,:) = KX.*KY;
M(1,3,:,:) = kappaPhibar;                    M(1,5,:,:) = -(1-phibar)*KX;
M(2,1,:,:) = KX.*KY;                         M(2,2,:,:) = -kappaPhibar + KX2+2*KY2;
M(2,4,:,:) = kappaPhibar;                    M(2,5,:,:) = -(1-phibar)*KY;
M(3,1,:,:) = kappaPhibar;
M(3,3,:,:) = -kappaPhibar + (etab/etaf)*(2*KX2+KY2); M(3,4,:,:) = (etab/etaf)*KX.*KY;
M(3,5,:,:) = -phibar*KX;
M(4,2,:,:) = kappaPhibar;                    M(4,3,:,:) = (etab/etaf)*KX.*KY;
M(4,4,:,:) = -kappaPhibar + (etab/etaf)*(KX2+2*KY2); M(4,5,:,:) = -phibar*KY;
M(5,1,:,:) = (1-phibar)*KX;  M(5,2,:,:) = (1-phibar)*KY;
M(5,3,:,:) = phibar*KX;      M(5,4,:,:) = phibar*KY;

M(1,:,1,1) = 0; M(1,1,1,1) = scaleVal;
M(2,:,1,1) = 0; M(2,2,1,1) = scaleVal;

M(5,:,1,1) = 0; M(:,5,1,1) = 0; M(5,5,1,1) = scaleVal;

matchedY = true(Ny,1); matchedY(Ny/2:Ny/2+2) = false;
matchedX = true(Nx,1); matchedX(Nx/2:Nx/2+2) = false;
pressureMatched = matchedY & matchedX';

[iyUnmatched, ixUnmatched] = find(~pressureMatched);
for k = 1:numel(iyUnmatched)
    iy = iyUnmatched(k); ix = ixUnmatched(k);
    M(5,:,iy,ix) = 0; M(:,5,iy,ix) = 0; M(5,5,iy,ix) = scaleVal;
end

Minv = pageinv(M);
end

function out = velocityAvScaled(v, ctx)
% Identical to velocityAv.m, except the gauge-pin rows are multiplied by
% ctx.gaugeScale instead of returning the bare mean -- see Part 3's header
% comment above.
Nx = ctx.Nx; Ny = ctx.Ny; Nfine = ctx.Nfine; Np = ctx.Np; NP = ctx.NP;
D = ctx.D; ksi_A2 = ctx.ksi_A2; phiext = ctx.phiext;
rowScale = ctx.rowScale; RefPtIDXfine = ctx.RefPtIDXfine;
etab = ctx.etab; etaf = ctx.etaf; KXY = ctx.KXY;
KXgrid = D.KX; KYgrid = D.KY; KX2grid = D.KX2; KY2grid = D.KY2;

dUf1 = v(1:Nfine); dUf2 = v(Nfine+1:2*Nfine);
dUp1 = v(2*Nfine+1:3*Nfine); dUp2 = v(3*Nfine+1:4*Nfine);
dP   = v(4*Nfine+1:4*Nfine+Np);
dPfull = [dP; -sum(dP)];

Fuf1 = fft2(reshape(dUf1, Ny, Nx)); Fuf2 = fft2(reshape(dUf2, Ny, Nx));
Fup1 = fft2(reshape(dUp1, Ny, Nx)); Fup2 = fft2(reshape(dUp2, Ny, Nx));
Fdp  = D.liftToFourier(dPfull);

toCol = @(A) reshape(A, Nfine, 1);
DXdp = toCol(real(ifft2(KXgrid.*Fdp)));
DYdp = toCol(real(ifft2(KYgrid.*Fdp)));

out1 = -ksi_A2.*dUf1 + toCol(real(ifft2((2*KX2grid+KY2grid).*Fuf1 + KXY.*Fuf2))) ...
     + ksi_A2.*dUp1 - (1-phiext).*DXdp;
out2 = -ksi_A2.*dUf2 + toCol(real(ifft2(KXY.*Fuf1 + (KX2grid+2*KY2grid).*Fuf2))) ...
     + ksi_A2.*dUp2 - (1-phiext).*DYdp;
out3 = ksi_A2.*dUf1 - ksi_A2.*dUp1 ...
     + (etab/etaf)*toCol(real(ifft2((2*KX2grid+KY2grid).*Fup1 + KXY.*Fup2))) ...
     - phiext.*DXdp;
out4 = ksi_A2.*dUf2 - ksi_A2.*dUp2 ...
     + (etab/etaf)*toCol(real(ifft2(KXY.*Fup1 + (KX2grid+2*KY2grid).*Fup2))) ...
     - phiext.*DYdp;

Fcombo_x = fft2(reshape((1-phiext).*dUf1 + phiext.*dUp1, Ny, Nx));
Fcombo_y = fft2(reshape((1-phiext).*dUf2 + phiext.*dUp2, Ny, Nx));
tmp5c = D.truncateFromFourier(KXgrid.*Fcombo_x + KYgrid.*Fcombo_y);
out5 = rowScale * (tmp5c(1:Np) - tmp5c(NP));

out1(RefPtIDXfine) = ctx.gaugeScale * mean(dUf1);
out2(RefPtIDXfine) = ctx.gaugeScale * mean(dUf2);

out = [out1; out2; out3; out4; out5];
end

%% ===================== local functions =====================
function ctx = buildCtxForTest(phi, parameter, D)
% Same construction as solveVelocityFFT.m's setup section (lines ~84-113),
% trimmed to exactly what velocityAv.m/velocityPinv.m need -- no RHS/theta
% terms, since those don't affect the OPERATOR (only the forcing), and the
% operator/preconditioner's conditioning is all this test is probing.
Nx = parameter.Nx; Ny = parameter.Ny;
etab = parameter.etab; etaf = parameter.etaf;
Nfine = Nx*Ny; NP = (Nx-2)*(Ny-2); Np = NP - 1;

phiext = D.liftP(phi);
phibar = mean(phi);
ksi_A2 = 2*parameter.A^2/parameter.R^2 * phiext;

Mscale = max(abs(ksi_A2));
kMax   = max(abs(D.KX(:)));
rowScale = Mscale / kMax;

RefPtIDXfine = Ny*(Nx/2 - 1) + Ny/2;

[Minv, Mblock] = buildVelocityPreconditioner(phibar, parameter, D);

ctx = struct('Nx', Nx, 'Ny', Ny, 'Nfine', Nfine, 'NP', NP, 'Np', Np, ...
    'D', D, 'ksi_A2', ksi_A2, 'phiext', phiext, 'rowScale', rowScale, ...
    'RefPtIDXfine', RefPtIDXfine, 'etab', etab, 'etaf', etaf, ...
    'Minv', Minv, 'M', Mblock, 'KXY', D.KX .* D.KY);
end

function c = maxCondOverModes(Mblock)
% Worst-conditioned single 5x5 per-mode block in buildVelocityPreconditioner.m's
% output (5 x 5 x Ny x Nx layout) -- cheap exact per-mode cond(), no dense
% n x n assembly needed since M is block-diagonal per mode by construction.
[~, ~, Ny, Nx] = size(Mblock);
c = 0;
for ix = 1:Nx
    for iy = 1:Ny
        cc = cond(Mblock(:,:,iy,ix));
        if cc > c; c = cc; end
    end
end
end
