function T = buildRestrictTruncMat(Nx, Ny)
%BUILDRESTRICTTRUNCMAT Dense Ncoarse x Nfine fine->coarse restriction
%   matrix via spectral TRUNCATION, as an alternative to
%   getInterpolationMatrix.m's interMat (which restricts via ALIASING --
%   collocation, evaluating the fine trigonometric interpolant at the
%   coarse points). This file is independent of interMat/interMatP --
%   nothing else is changed; it exists as an alternative to swap in where
%   wanted, e.g. for a continuity-equation discretization that only
%   constrains the Fourier modes pressure can actually represent.
%
%   The two are NOT the same operator: interMat treats every fine mode as
%   if it belongs to some coarse mode (aliasing/folding higher frequencies
%   in, since Nc<Nf forces every fine sample's information into the
%   coarse count somehow); this instead DISCARDS every frequency above
%   what the coarse grid can represent, keeping only the low-frequency
%   content -- the true adjoint of liftCoarseToFine's spectral zero-
%   padding (conjugate the same grid-offset phase correction, and at the
%   Nyquist bin, COMBINE the fine grid's separate +Nyquist/-Nyquist
%   values instead of SPLITTING one coarse value into them).
%
%   Verified: truncate(lift(x)) = x to machine precision (recovers exactly
%   what was embedded, same as interMat's own aliasing round-trip
%   property), and truncate(pure coarse-unresolvable frequency) = exactly
%   0 (discarded, not folded in -- unlike interMat, which gives a nonzero
%   result for the same input since it aliases that content elsewhere).
%
%   Built via unit-vector probing of the verified FFT-based truncation
%   operator (truncateUImpl below) -- a one-time O(Nfine) cost of small
%   FFTs, the same style already used for interMat/interMatP.

Nfine = Nx*Ny;
NyC = Ny-2; NxC = Nx-2;
Ncoarse = NxC*NyC;

T = zeros(Ncoarse, Nfine);
for j = 1:Nfine
    e = zeros(Nfine,1);
    e(j) = 1;
    T(:,j) = truncateUImpl(e, Ny, Nx);
end
end

function xc = truncateUImpl(xf, Ny, Nx)
% Self-contained 2D spectral truncation: fine NyxNx -> coarse
% (Ny-2)x(Nx-2), keeping only the low-frequency content the coarse grid
% can represent and discarding the rest (as opposed to an aliasing restrict
% like getInterpolationMatrix.m's interMat, which ALIASES/folds the rest in
% instead of discarding it). Truncation is separable and composes correctly axis-by-axis
% directly in frequency space -- no real-space pass needed in between
% (unlike aliasing, where the two 1D fold operations don't compose
% correctly back-to-back in frequency space).
Ax = 1; L = 2*Ax;
NyC = Ny-2; NxC = Nx-2;
Xf = fft2(reshape(xf, Ny, Nx));

%% Truncate dim 2 (x): Nx -> Nx-2, via transpose.
XfT = Xf.';
[Nf1, M1] = size(XfT);
half1 = NxC/2;
shift1 = 2*Ax*(1/Nf1 - 1/NxC);
tmpT = zeros(NxC, M1);
k_pos1 = (0:half1-1)';
tmpT(1:half1,:) = XfT(1:half1,:) .* exp(-1i*2*pi*k_pos1*shift1/L);
phase_nyq1 = exp(1i*2*pi*half1*shift1/L);
% Adjoint of the pad's Nyquist SPLIT (one coarse value -> two fine bins)
% is a COMBINE here: recombine the fine grid's separate +Nyquist/-Nyquist
% values back into the single coarse Nyquist value.
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
