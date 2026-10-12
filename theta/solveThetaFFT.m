function theta_new = solveThetaFFT(theta_old, phi, Uf1, Uf2, Up1, Up2, dt_step, parameter, D)
% Matrix-free, FFT-based twin of solveTheta.m -- same Backward Euler
% discretization, just applied without ever assembling solveTheta.m's
% dense Nfine x Nfine matrix. Solved with GMRES, RIGHT-preconditioned (see
% solveVelocityFFT.m for why right, not left: its relres is the TRUE
% residual, not a preconditioned stand-in) by the same operator with phi
% frozen to its mean (so its Laplacian term is exactly constant-
% coefficient) and Up1,Up2 frozen to their means.
%
% Warm-started from theta_old: under right preconditioning GMRES solves
% for y where x=thetaPinv(y), so passing theta_old as x0 directly would be
% wrong -- it needs mapping through thetaPinv's FORWARD (uninverted)
% operator first, so that thetaPinv(y0)~=theta_old. thetaPinv is exactly
% "divide by invDenom in Fourier space" (no lift/truncate -- theta already
% lives on the fine grid), so its exact forward map is "multiply by
% invDenom" instead, i.e. divide by thetaPinv's OWN divisor.
%
% D = fourierDiffOps(Nx,Ny), the same D the caller already builds once for
% solveVelocityFFT.m/solvePhiFFT.m -- Nx,Ny fixed for the whole run,
% passed in rather than recomputed here. phi is on the COARSE grid (same
% as solveTheta.m's CoarseMesh.interMatP*phi); D.liftP(phi) is its
% FFT-based equivalent (already the pattern used by solveVelocityFFT.m/
% computeVelocityRHS.m for the same lift).

Nx = parameter.Nx; Ny = parameter.Ny;
Nfine = Nx*Ny;
A_param  = parameter.A;
V0       = parameter.V0;
etaf     = parameter.etaf;
R        = parameter.R;
L_param  = parameter.L;
elasPara = parameter.elasPara;
ksif     = parameter.ksif;

c0 = elasPara * 6*R^2 / (etaf * L_param^2);
coeff = c0/(V0*A_param);

phiext = D.liftP(phi);
DXphiext = D.DXf(phiext);
DYphiext = D.DYf(phiext);

%% Frozen-mean preconditioner: phi frozen to phibar (zeroing DXphiext/
% DYphiext's role there) and Up1,Up2 frozen to their means makes the
% operator constant-coefficient, hence a per-mode scalar in Fourier space.
phibar = mean(phiext);
a = mean(Up1); b = mean(Up2);
invDenom = 1 ./ (1/dt_step + a*D.KX + b*D.KY - coeff*phibar*(D.KX2 + D.KY2));

ctx = struct('D', D, 'Up1', Up1, 'Up2', Up2, 'phiext', phiext, ...
    'DXphiext', DXphiext, 'DYphiext', DYphiext, 'coeff', coeff, ...
    'dt_step', dt_step, 'Nx', Nx, 'Ny', Ny, 'invDenom', invDenom);

rhs = computeThetaRHS(theta_old, Uf1, Uf2, Up1, Up2, dt_step, ksif, D);

y0 = real(ifft2(fft2(reshape(theta_old, Ny, Nx)) ./ invDenom));
y0 = y0(:);

tol = parameter.gmresTol;      % single shared value, set once in mainPC.m -- see its header comment
maxit = parameter.gmresMaxIt;  % single shared value, set once in mainPC.m -- see its header comment
AvPinv = @(y) thetaAv(thetaPinv(y, ctx), ctx);
[y, gmresFlag, gmresRelres, gmresIter] = gmres(AvPinv, rhs, min(parameter.gmresRestartScalar,Nfine), tol, maxit, [], [], y0);
theta_new = thetaPinv(y, ctx);

fprintf('  solveThetaFFT: gmres flag=%d relres=%.2e iters=%d/%d\n', ...
    gmresFlag, gmresRelres, gmresIter(1), gmresIter(2));
end
