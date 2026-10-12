function phi_new = solvePhiFFT(phi_old, Up1, Up2, dt_step, parameter, D)
% Matrix-free, FFT-based twin of solvePhi.m -- same Backward Euler
% discretization of d(phi)/dt + div(phi*Up) = 0, same interMatP/interMat
% (coarse<->fine) projections, just applied without ever assembling
% solvePhi.m's dense Ncoarse x Ncoarse matrix. Solved with GMRES, RIGHT-
% preconditioned (see solveVelocityFFT.m for why right, not left: its
% relres is the TRUE residual, not a preconditioned stand-in) by the same
% operator with Up1,Up2 frozen to their spatial means.
%
% rhs simplification: solvePhi.m's rhs = interMat*interMatP*phi_old/dt --
% since interMat is the TRUNCATION restrict (buildRestrictTruncMat) and
% interMatP is the spectral lift, interMat*interMatP = I exactly (verified
% elsewhere: truncate(lift(x))=x to machine precision), so this collapses
% to phi_old/dt_step directly -- no matrix-vector product needed.
%
% Warm-started from phi_old: under right preconditioning GMRES solves for
% y where x=phiPinv(y), so phi_old itself would be the wrong thing to hand
% GMRES as x0 -- it needs mapping through phiPinv's FORWARD (uninverted)
% operator first, so that phiPinv(y0)~=phi_old. phiPinv sandwiches its
% divide-by-invDenom between D.liftToFourier and D.truncateFromFourier,
% and lift-then-truncate is NOT the identity (only truncate-then-lift is),
% so this mapping is only APPROXIMATE -- exact at every mode except the
% Nyquist-band flanking bins (same caveat as buildVelocityPreconditioner.m's
% header discusses for a different operator). Fine for a warm start, which
% only needs to be closer than zero, not exact.
%
% D = fourierDiffOps(Nx,Ny), the same D the caller already builds once for
% solveVelocityFFT.m -- Nx,Ny fixed for the whole run, passed in rather
% than recomputed here.

Nx = parameter.Nx; Ny = parameter.Ny;
Ncoarse = numel(phi_old);

divUp = D.DXf(Up1) + D.DYf(Up2);

%% Frozen-mean preconditioner: constant-coefficient advection is exactly
% diagonal per Fourier mode, eigenvalue a*KX+b*KY+c (a,b,c = spatial means
% of Up1,Up2,divUp). No decoupling/rank-deficiency concerns here (unlike
% buildVelocityPreconditioner.m's pressure block): 1/dt_step alone keeps
% every mode's denominator nonzero even where a*KX+b*KY+c vanishes.
a = mean(Up1); b = mean(Up2); c = mean(divUp);
invDenom = 1 ./ (1/dt_step + a*D.KX + b*D.KY + c);

ctx = struct('D', D, 'Up1', Up1, 'Up2', Up2, 'divUp', divUp, ...
    'dt_step', dt_step, 'Nx', Nx, 'Ny', Ny, 'invDenom', invDenom);

rhs = phi_old / dt_step;

y0 = D.truncateFromFourier(D.liftToFourier(phi_old) .* (1./invDenom));

tol = parameter.gmresTol;      % single shared value, set once in mainPC.m -- see its header comment
maxit = parameter.gmresMaxIt;  % single shared value, set once in mainPC.m -- see its header comment
AvPinv = @(y) phiAv(phiPinv(y, ctx), ctx);
[y, gmresFlag, gmresRelres, gmresIter] = gmres(AvPinv, rhs, min(parameter.gmresRestartScalar,Ncoarse), tol, maxit, [], [], y0);
phi_new = phiPinv(y, ctx);

fprintf('  solvePhiFFT: gmres flag=%d relres=%.2e iters=%d/%d\n', ...
    gmresFlag, gmresRelres, gmresIter(1), gmresIter(2));
end
