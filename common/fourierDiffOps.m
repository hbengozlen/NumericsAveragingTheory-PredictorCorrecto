function D = fourierDiffOps(Nx, Ny)
%FOURIERDIFFOPS Spectral differentiation/lift/restrict operators for an
%   Nx x Ny periodic fine grid paired with its (Nx-2) x (Ny-2) coarse grid.
%
%   Returns a struct of function handles -- D.DXf, D.DYf, D.D2Xf, D.D2Yf
%   (fine-grid derivatives), D.liftP (coarse->fine, real space in and out),
%   D.liftToFourier (same lift, but skips a redundant ifft2-then-fft2 round
%   trip when the caller only ever needs the lifted field's OWN Fourier
%   transform), D.truncateFromFourier (fine->coarse restriction via
%   spectral TRUNCATION -- discards fine-grid content the coarse grid
%   can't represent instead of folding it into a representable mode; the
%   true adjoint of liftToFourier's zero-padding, starting directly from
%   Fourier space) -- plus the raw eigenvalue grids D.KX, D.KY, D.KX2,
%   D.KY2 (Ny x Nx).
%
%   Each of these four is fully SELF-CONTAINED below: the spectral
%   zero-padding (coarse->fine) and aliasing/truncation (fine->coarse)
%   math is written out directly inside each one, rather than any of them
%   calling liftCoarseToFine.m/restrictFineToCoarse.m/fourierUpsample2D.m/
%   fourierPad1D.m/fourierTrunc1D.m, or even sharing a helper with each
%   other -- so reading any ONE of the four shows its whole computation,
%   with nothing to chase elsewhere. This does mean the same pad/truncate
%   arithmetic appears more than once below (once per function that needs
%   it); that repetition is intentional, not an oversight. (The original
%   separate files are untouched and still used by solveVelocityFFT_schur.m.)
%
%   D2X/D2Y come from fourierGrid2's DIRECT second-derivative matrix, NOT
%   KX.^2/KY.^2 -- KX is zero at the Nyquist mode (its derivative direction
%   is ambiguous on an even grid), and squaring it inherits that zero. A
%   direct second derivative has no such ambiguity and gives the correct
%   nonzero curvature there instead.
%
%   Pure function -- no internal caching. Nx,Ny are fixed for an entire
%   simulation run, so callers should call this ONCE at setup (e.g.
%   mainPC.m, right where parameter.DX/DY/D2X/D2Y are already built) and
%   pass the resulting D down to every solver that needs it, rather than
%   having each solver call this itself every timestep.

[~, Dx1] = fourierGrid(1, Nx);
[~, Dy1] = fourierGrid(1, Ny);
lambda_x = fft(Dx1(:,1));
lambda_y = fft(Dy1(:,1));
[KXgrid, KYgrid] = meshgrid(lambda_x, lambda_y);   % Ny x Nx
[~, D2x1] = fourierGrid2(1, Nx);
[~, D2y1] = fourierGrid2(1, Ny);
lambda2_x = fft(D2x1(:,1));
lambda2_y = fft(D2y1(:,1));
[KX2grid, KY2grid] = meshgrid(lambda2_x, lambda2_y);

D.KX  = KXgrid;
D.KY  = KYgrid;
D.KX2 = KX2grid;
D.KY2 = KY2grid;
D.Nx  = Nx;
D.Ny  = Ny;

Nfine = Nx*Ny;
D.DXf  = @(x) reshape(real(ifft2(KXgrid  .* fft2(reshape(x, Ny, Nx)))), Nfine, 1);
D.DYf  = @(x) reshape(real(ifft2(KYgrid  .* fft2(reshape(x, Ny, Nx)))), Nfine, 1);
D.D2Xf = @(x) reshape(real(ifft2(KX2grid .* fft2(reshape(x, Ny, Nx)))), Nfine, 1);
D.D2Yf = @(x) reshape(real(ifft2(KY2grid .* fft2(reshape(x, Ny, Nx)))), Nfine, 1);

D.liftP               = @(x)  liftPImpl(x, Ny, Nx);
D.liftToFourier       = @(xc) liftToFourierImpl(xc, Ny, Nx);
D.truncateFromFourier = @(Xf) truncateFromFourierImpl(Xf, Ny, Nx);
end

function xf = liftPImpl(xc, Ny, Nx)
%LIFTPIMPL Self-contained coarse (Ny-2)x(Nx-2) -> fine NyxNx spectral
%   lift, real space in and out. Ax=1 throughout (this codebase's fixed
%   convention -- see fourierGrid(1,N) everywhere).
Ax = 1; L = 2*Ax;
Xc = fft2(reshape(xc, Ny-2, Nx-2));

%% Pad dim 1 (y): Ny-2 -> Ny. Two corrections beyond naive zero-padding:
% (1) the ambiguous coarse Nyquist bin is SPLIT in half between the fine
% grid's +Nyquist/-Nyquist bins (a real Nyquist mode has no phase freedom
% on the coarse grid, but the fine grid can tell them apart); (2) a
% per-mode linear phase correction, since fourierGrid(1,Nc) and
% fourierGrid(1,Nf) place their first grid point at slightly different
% physical x (their spacings 2/Nc, 2/Nf differ).
[Nc1, M1] = size(Xc);
half1 = Nc1/2;
shift1 = 2*Ax*(1/Ny - 1/Nc1);
tmp = zeros(Ny, M1);
k_pos1 = (0:half1-1)';
tmp(1:half1,:) = Xc(1:half1,:) .* exp(1i*2*pi*k_pos1*shift1/L);
phase_nyq1 = exp(1i*2*pi*half1*shift1/L);
tmp(half1+1,:)        = Xc(half1+1,:)/2 * phase_nyq1;
tmp(Ny-half1+1,:)     = tmp(Ny-half1+1,:) + Xc(half1+1,:)/2 * conj(phase_nyq1);
k_neg1 = (-(half1-1):-1)';
tmp(Ny-half1+2:Ny,:)  = Xc(half1+2:Nc1,:) .* exp(1i*2*pi*k_neg1*shift1/L);
tmp = tmp * (Ny/Nc1);

%% Pad dim 2 (x): Nx-2 -> Nx, via transpose (same math, other axis).
tmpT = tmp.';
[Nc2, M2] = size(tmpT);
half2 = Nc2/2;
shift2 = 2*Ax*(1/Nx - 1/Nc2);
XfT = zeros(Nx, M2);
k_pos2 = (0:half2-1)';
XfT(1:half2,:) = tmpT(1:half2,:) .* exp(1i*2*pi*k_pos2*shift2/L);
phase_nyq2 = exp(1i*2*pi*half2*shift2/L);
XfT(half2+1,:)        = tmpT(half2+1,:)/2 * phase_nyq2;
XfT(Nx-half2+1,:)     = XfT(Nx-half2+1,:) + tmpT(half2+1,:)/2 * conj(phase_nyq2);
k_neg2 = (-(half2-1):-1)';
XfT(Nx-half2+2:Nx,:)  = tmpT(half2+2:Nc2,:) .* exp(1i*2*pi*k_neg2*shift2/L);
XfT = XfT * (Nx/Nc2);
Xf = XfT.';

xf = real(ifft2(Xf));
xf = xf(:);
end

function Xf = liftToFourierImpl(xc, Ny, Nx)
%LIFTTOFOURIERIMPL Same result as fft2(reshape(liftPImpl(xc,Ny,Nx),Ny,Nx)),
%   but stops right after the zero-padding (skips the final ifft2, and
%   thus the fft2 the caller would otherwise redo on the way back in) --
%   for callers that only ever need the lifted field's OWN Fourier
%   transform, not its real-space values. Self-contained: same pad math as
%   liftPImpl, repeated here rather than shared, so this function alone
%   shows its whole computation.
Ax = 1; L = 2*Ax;
Xc = fft2(reshape(xc, Ny-2, Nx-2));

%% Pad dim 1 (y): Ny-2 -> Ny.
[Nc1, M1] = size(Xc);
half1 = Nc1/2;
shift1 = 2*Ax*(1/Ny - 1/Nc1);
tmp = zeros(Ny, M1);
k_pos1 = (0:half1-1)';
tmp(1:half1,:) = Xc(1:half1,:) .* exp(1i*2*pi*k_pos1*shift1/L);
phase_nyq1 = exp(1i*2*pi*half1*shift1/L);
tmp(half1+1,:)        = Xc(half1+1,:)/2 * phase_nyq1;
tmp(Ny-half1+1,:)     = tmp(Ny-half1+1,:) + Xc(half1+1,:)/2 * conj(phase_nyq1);
k_neg1 = (-(half1-1):-1)';
tmp(Ny-half1+2:Ny,:)  = Xc(half1+2:Nc1,:) .* exp(1i*2*pi*k_neg1*shift1/L);
tmp = tmp * (Ny/Nc1);

%% Pad dim 2 (x): Nx-2 -> Nx, via transpose.
tmpT = tmp.';
[Nc2, M2] = size(tmpT);
half2 = Nc2/2;
shift2 = 2*Ax*(1/Nx - 1/Nc2);
XfT = zeros(Nx, M2);
k_pos2 = (0:half2-1)';
XfT(1:half2,:) = tmpT(1:half2,:) .* exp(1i*2*pi*k_pos2*shift2/L);
phase_nyq2 = exp(1i*2*pi*half2*shift2/L);
XfT(half2+1,:)        = tmpT(half2+1,:)/2 * phase_nyq2;
XfT(Nx-half2+1,:)     = XfT(Nx-half2+1,:) + tmpT(half2+1,:)/2 * conj(phase_nyq2);
k_neg2 = (-(half2-1):-1)';
XfT(Nx-half2+2:Nx,:)  = tmpT(half2+2:Nc2,:) .* exp(1i*2*pi*k_neg2*shift2/L);
XfT = XfT * (Nx/Nc2);
Xf = XfT.';
end

function xc = truncateFromFourierImpl(Xf, Ny, Nx)
%TRUNCATEFROMFOURIERIMPL Fine (Ny x Nx) -> coarse (Ny-2)x(Nx-2) restriction
%   via spectral TRUNCATION, starting directly from an already-Fourier-
%   space Xf (skips a redundant fft2 for callers that already have the
%   fine field's Fourier transform in hand and don't need its real-space
%   values). This is the matrix-free, FFT-based twin of
%   buildRestrictTruncMat.m (used by solveVelocity.m via
%   FineMesh.interMatU): same operator, verified equivalent, just computed
%   via O(N log N) FFT truncation instead of an O(Ncoarse*Nfine) dense
%   matrix-vector product.
%
%   NOT an aliasing restrict (which folds frequencies above the coarse
%   Nyquist limit into a representable one, e.g. getInterpolationMatrix.m's
%   interMat). This instead DISCARDS them: it's the true adjoint of
%   liftToFourierImpl's spectral zero-padding -- conjugate the same
%   grid-offset phase correction, and at the Nyquist bin, COMBINE the fine
%   grid's separate +Nyquist/-Nyquist values back into one (instead of
%   aliasing splitting one coarse value into them). Self-contained: no
%   shared helper with buildRestrictTruncMat, so this one function shows
%   its whole computation.
Ax = 1; L = 2*Ax;
NyC = Ny-2; NxC = Nx-2;

%% Truncate dim 2 (x): Nx -> Nx-2, via transpose.
XfT = Xf.';
[Nf1, M1] = size(XfT);
half1 = NxC/2;
shift1 = 2*Ax*(1/Nf1 - 1/NxC);
tmpT = zeros(NxC, M1);
k_pos1 = (0:half1-1)';
tmpT(1:half1,:) = XfT(1:half1,:) .* exp(-1i*2*pi*k_pos1*shift1/L);
phase_nyq1 = exp(1i*2*pi*half1*shift1/L);
tmpT(half1+1,:) = XfT(half1+1,:)*conj(phase_nyq1) + XfT(Nf1-half1+1,:)*phase_nyq1;
k_neg1 = (-(half1-1):-1)';
tmpT(half1+2:NxC,:) = XfT(Nf1-half1+2:Nf1,:) .* exp(-1i*2*pi*k_neg1*shift1/L);
tmpT = tmpT * (NxC/Nf1);
tmp = tmpT.';

%% Truncate dim 1 (y): Ny -> Ny-2.
[Nf2, M2] = size(tmp);
half2 = NyC/2;
shift2 = 2*Ax*(1/Nf2 - 1/NyC);
Xc = zeros(NyC, M2);
k_pos2 = (0:half2-1)';
Xc(1:half2,:) = tmp(1:half2,:) .* exp(-1i*2*pi*k_pos2*shift2/L);
phase_nyq2 = exp(1i*2*pi*half2*shift2/L);
Xc(half2+1,:) = tmp(half2+1,:)*conj(phase_nyq2) + tmp(Nf2-half2+1,:)*phase_nyq2;
k_neg2 = (-(half2-1):-1)';
Xc(half2+2:NyC,:) = tmp(Nf2-half2+2:Nf2,:) .* exp(-1i*2*pi*k_neg2*shift2/L);
Xc = Xc * (NyC/Nf2);

xc = real(ifft2(Xc));
xc = xc(:);
end
