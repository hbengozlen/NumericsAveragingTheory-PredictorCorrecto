function out = velocityAv(v, ctx)
% Matrix-free operator for solveVelocityFFT.m: velocityAv(v) plays the
% role of A_sys*v, using the TRUE (spatially-varying phi) coefficients --
% exact, not an approximation (verified elsewhere to match the dense
% A_sys to machine precision). Named velocityAv (not the generic "Av") so
% it doesn't collide with a future solveThetaFFT/solvePhiFFT's own
% matrix-free operator on the same MATLAB path.
tAv = tic;
Nx = ctx.Nx; Ny = ctx.Ny; Nfine = ctx.Nfine; Np = ctx.Np; NP = ctx.NP;
D = ctx.D; ksi_A2 = ctx.ksi_A2; phiext = ctx.phiext;
rowScale = ctx.rowScale; RefPtIDXfine = ctx.RefPtIDXfine;
etab = ctx.etab; etaf = ctx.etaf; KXY = ctx.KXY;
KXgrid = D.KX; KYgrid = D.KY; KX2grid = D.KX2; KY2grid = D.KY2;

dUf1 = v(1:Nfine);
dUf2 = v(Nfine+1:2*Nfine);
dUp1 = v(2*Nfine+1:3*Nfine);
dUp2 = v(3*Nfine+1:4*Nfine);
dP   = v(4*Nfine+1:4*Nfine+Np);

dPfull = [dP; -sum(dP)];

% Transform each real-space field ONCE and derive every needed derivative
% by combining eigenvalues in Fourier space, instead of each DXf/DYf/D2Xf/
% D2Yf call independently re-transforming the same input: e.g.
% 2*D2Xf(dUf1)+D2Yf(dUf1) would compute fft2(dUf1) twice, and DYf(DXf(x))
% would round-trip through real space in the middle for no reason, since
% KX,KY commute as elementwise multiplies in Fourier space -- combining
% them first is also slightly MORE accurate, not just faster, since it
% skips an intermediate real-space rounding step (same reasoning as why
% D2X/D2Y are direct second-derivative matrices, not DX*DX). dP_fine's
% Fourier transform (Fdp) is obtained directly from dPfull via
% liftToFourier, skipping the real-space round trip entirely -- dP_fine
% itself was never actually used in real space, only its derivatives were.
Fuf1 = fft2(reshape(dUf1, Ny, Nx));
Fuf2 = fft2(reshape(dUf2, Ny, Nx));
Fup1 = fft2(reshape(dUp1, Ny, Nx));
Fup2 = fft2(reshape(dUp2, Ny, Nx));
Fdp  = D.liftToFourier(dPfull);

toCol = @(A) reshape(A, Nfine, 1);

DXdp = toCol(real(ifft2(KXgrid.*Fdp)));
DYdp = toCol(real(ifft2(KYgrid.*Fdp)));

% Row 1: fluid momentum X  (~ B11..B15)
out1 = -ksi_A2.*dUf1 + toCol(real(ifft2((2*KX2grid+KY2grid).*Fuf1 + KXY.*Fuf2))) ...
     + ksi_A2.*dUp1 - (1-phiext).*DXdp;

% Row 2: fluid momentum Y  (~ B21..B25)
out2 = -ksi_A2.*dUf2 + toCol(real(ifft2(KXY.*Fuf1 + (KX2grid+2*KY2grid).*Fuf2))) ...
     + ksi_A2.*dUp2 - (1-phiext).*DYdp;

% Row 3: particle momentum X  (~ B31..B35)
out3 = ksi_A2.*dUf1 - ksi_A2.*dUp1 ...
     + (etab/etaf)*toCol(real(ifft2((2*KX2grid+KY2grid).*Fup1 + KXY.*Fup2))) ...
     - phiext.*DXdp;

% Row 4: particle momentum Y  (~ B41..B45)
out4 = ksi_A2.*dUf2 - ksi_A2.*dUp2 ...
     + (etab/etaf)*toCol(real(ifft2(KXY.*Fup1 + (KX2grid+2*KY2grid).*Fup2))) ...
     - phiext.*DYdp;

% Row 5: fluid continuity  (~ B51..B54), row-scaled to match momentum rows.
% D.truncateFromFourier restricts fine->coarse via TRUNCATION (matches
% solveVelocity.m's buildRestrictTruncMat, wired in via FineMesh.interMatU)
% instead of an ALIASING restrict (e.g. interMat) -- only the divergence's
% low-frequency content, the part pressure can actually represent, enters
% the continuity constraint; content above the coarse Nyquist limit is
% discarded instead of being folded into some representable mode. Also
% skips a redundant fft2 by taking the already-computed Fourier-space sum
% directly.
Fcombo_x = fft2(reshape((1-phiext).*dUf1 + phiext.*dUp1, Ny, Nx));
Fcombo_y = fft2(reshape((1-phiext).*dUf2 + phiext.*dUp2, Ny, Nx));
tmp5c = D.truncateFromFourier(KXgrid.*Fcombo_x + KYgrid.*Fcombo_y);
out5 =rowScale * (tmp5c(1:Np) - tmp5c(NP));

% Velocity gauge pin: mean(Uf1)=0, mean(Uf2)=0 instead of a physical point
% pin. fft2(x)(1,1)=sum(x), so this is EXACTLY the DC-mode Fourier
% coefficient of Uf1/Uf2 -- a per-mode constraint the preconditioner can
% represent exactly (a physical point-pin spreads across every Fourier
% mode when transformed; this doesn't). RefPtIDXfine is just the arbitrary
% slot carrying this equation -- any index works, since mean() isn't tied
% to a physical location.
out1(RefPtIDXfine) = mean(dUf1);
out2(RefPtIDXfine) = mean(dUf2);

out = [out1; out2; out3; out4; out5];

perfCounter('solveVelocityFFT', 'add', 'Av', toc(tAv));
perfCounter('solveVelocityFFT', 'add', 'AvCalls', 1);
end
