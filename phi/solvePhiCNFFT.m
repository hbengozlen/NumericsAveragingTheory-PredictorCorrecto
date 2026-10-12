function phi_new = solvePhiCNFFT(phi_old, Up1, Up2, dt_step, parameter, D)
% Matrix-free, FFT-based twin of solvePhiCN.m -- same Crank-Nicolson
% discretization of d(phi)/dt + div(phi*Up) = 0, same interMatP/interMat
% (coarse<->fine) projections, just applied without ever assembling
% solvePhiCN.m's dense Ncoarse x Ncoarse matrix. Solved with GMRES, RIGHT-
% preconditioned (see solveVelocityFFT.m for why right, not left: its
% relres is the TRUE residual, not a preconditioned stand-in) by the same
% operator with Up1,Up2 frozen to their spatial means.
%
% Warm-started from phi_old via phiPinvCN's forward (uninverted) operator
% -- see solvePhiFFT.m's comment for why (same reasoning and the same
% Nyquist-band-flanking-bin approximation, CN's invDenomCN in place of
% BE's invDenom).
%
% D = fourierDiffOps(Nx,Ny), the same D the caller already builds once for
% solveVelocityFFT.m/solvePhiFFT.m -- Nx,Ny fixed for the whole run,
% passed in rather than recomputed here.

Nx = parameter.Nx; Ny = parameter.Ny;
Ncoarse = numel(phi_old);

divUp = D.DXf(Up1) + D.DYf(Up2);

%% Frozen-mean preconditioner: same reasoning as solvePhiFFT.m's, just
% with the extra 0.5 factor Crank-Nicolson puts on the implicit operator.
a = mean(Up1); b = mean(Up2); c = mean(divUp);
invDenomCN = 1 ./ (1/dt_step + 0.5*(a*D.KX + b*D.KY + c));

ctx = struct('D', D, 'Up1', Up1, 'Up2', Up2, 'divUp', divUp, ...
    'dt_step', dt_step, 'Nx', Nx, 'Ny', Ny, 'invDenomCN', invDenomCN);

rhs = computePhiCNRHS(phi_old, ctx);

y0 = D.truncateFromFourier(D.liftToFourier(phi_old) .* (1./invDenomCN));

tol = parameter.gmresTol;      % single shared value, set once in mainPC.m -- see its header comment
maxit = parameter.gmresMaxIt;  % single shared value, set once in mainPC.m -- see its header comment
AvPinv = @(y) phiAvCN(phiPinvCN(y, ctx), ctx);
[y, gmresFlag, gmresRelres, gmresIter] = gmres(AvPinv, rhs, min(parameter.gmresRestartScalar,Ncoarse), tol, maxit, [], [], y0);
phi_new = phiPinvCN(y, ctx);

fprintf('  solvePhiCNFFT: gmres flag=%d relres=%.2e iters=%d/%d\n', ...
    gmresFlag, gmresRelres, gmresIter(1), gmresIter(2));
end
