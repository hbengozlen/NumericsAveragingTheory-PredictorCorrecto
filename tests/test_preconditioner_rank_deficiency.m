%% test_preconditioner_rank_deficiency.m
% Checks rank(M) of buildVelocityPreconditioner.m's frozen-phibar,
% per-Fourier-mode 5x5 block M, EXHAUSTIVELY across every single (kx,ky)
% mode -- not just a few named corners, since a partial/incorrect edit to
% the decoupling logic could pass a corners-only check while still leaving
% some other mode rank-deficient (a real regression caught here once: a
% partial revert left DC's pressure decoupling out entirely).
%
% For the actual dense-vs-matrix-free ACCURACY comparison (which this rank
% check alone does NOT guarantee -- a matrix can be full rank yet still
% catastrophically ill-conditioned, or the decoupling mask can be "matched"
% in a way that still misses real cross-mode coupling), see
% test_velocity_fft_accuracy.m instead. Both failure modes were found
% there, not here: rank(M)=5/5 everywhere in both cases, yet:
%   - Option A's decoupling (4 corner modes only) left cond(M)~1e6 at
%     non-corner Nyquist-adjacent modes, causing real GMRES stagnation and
%     10-40x error at larger theta amplitude.
%   - narrowing Option B's gap band to just the single true-Nyquist index
%     (leaving its 2 flanking bins "matched") reintroduced the same error
%     for a different reason: those flanking bins are the fine grid's two
%     halves of ONE coarse-Nyquist pressure dof, a cross-mode coupling this
%     per-mode-block-diagonal preconditioner can never represent, matched
%     or not.
% Rank alone caught neither. Run both tests, not just this one.

clear; clc;
addpath('../common');   % fourierGrid*.m, fourierDiffOps.m, etc.

%% Test case setup (small grid, fast to run) -- only what buildVelocityPreconditioner.m's
%% underlying M actually needs: parameter.{Nx,Ny,A,R,etab,etaf} and D.
parameter.A    = 200e-6;
parameter.etaf = 1e-3;
parameter.etab = 1e-4;
parameter.R    = 0.7e-6;

Nx = 10; Ny = 10;
parameter.Nx = Nx; parameter.Ny = Ny;

D = fourierDiffOps(Nx, Ny);

Ncoarse = (Nx-2)*(Ny-2);
rng(7);
phi = 0.15 + 0.03*randn(Ncoarse,1);

%% Exhaustive rank check across EVERY mode
fprintf('=== Exhaustive rank(M) check across every (kx,ky) mode ===\n\n');

phibar      = mean(phi);
kappaPhibar = 2*parameter.A^2/parameter.R^2*phibar;
etab = parameter.etab; etaf = parameter.etaf;

KXgrid = D.KX; KYgrid = D.KY; KX2grid = D.KX2; KY2grid = D.KY2;

% Same mask as buildVelocityPreconditioner.m's currently ACTIVE option
% (Option B, the 3-wide gap band per axis).
matchedY = true(Ny,1); matchedY(Ny/2:Ny/2+2) = false;
matchedX = true(Nx,1); matchedX(Nx/2:Nx/2+2) = false;
pressureMatched = matchedY & matchedX';   % Ny x Nx logical

buildM = @(KX,KY,KX2,KY2) [ ...
    -kappaPhibar+2*KX2+KY2,  KX*KY,                   kappaPhibar,                          0,                                    -(1-phibar)*KX; ...
    KX*KY,                   -kappaPhibar+KX2+2*KY2,  0,                                    kappaPhibar,                          -(1-phibar)*KY; ...
    kappaPhibar,              0,                      -kappaPhibar+(etab/etaf)*(2*KX2+KY2), (etab/etaf)*KX*KY,                    -phibar*KX; ...
    0,                        kappaPhibar,             (etab/etaf)*KX*KY,                    -kappaPhibar+(etab/etaf)*(KX2+2*KY2), -phibar*KY; ...
    (1-phibar)*KX,            (1-phibar)*KY,           phibar*KX,                            phibar*KY,                            0];

badCount = 0;
for ix = 1:Nx
    for iy = 1:Ny
        M = buildM(KXgrid(iy,ix), KYgrid(iy,ix), KX2grid(iy,ix), KY2grid(iy,ix));
        if iy == 1 && ix == 1
            M(1,:) = 0; M(1,1) = 1;
            M(2,:) = 0; M(2,2) = 1;
            M(5,:) = 0; M(:,5) = 0; M(5,5) = 1;
        elseif ~pressureMatched(iy,ix)
            M(5,:) = 0; M(:,5) = 0; M(5,5) = 1;
        end
        if rank(M) < 5
            badCount = badCount + 1;
        end
    end
end
fprintf('Exhaustive check: %d / %d modes are rank-deficient (should be 0)\n', badCount, Nx*Ny);
