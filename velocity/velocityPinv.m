function w = velocityPinv(r, ctx)
% Applies solveVelocityFFT.m's frozen-phibar preconditioner (ctx.Minv,
% from buildVelocityPreconditioner.m): FFT each residual component,
% multiply by the precomputed per-mode 5x5 inverse (vectorized across all
% modes at once), IFFT back. Named velocityPinv (not the generic "Pinv")
% so it doesn't collide with a future solveThetaFFT/solvePhiFFT's own
% preconditioner-apply function on the same MATLAB path.
tPinv = tic;
Nx = ctx.Nx; Ny = ctx.Ny; Nfine = ctx.Nfine; Np = ctx.Np; NP = ctx.NP;
D = ctx.D; Minv = ctx.Minv;

r1 = r(1:Nfine); r2 = r(Nfine+1:2*Nfine);
r3 = r(2*Nfine+1:3*Nfine); r4 = r(3*Nfine+1:4*Nfine);
r5 = r(4*Nfine+1:4*Nfine+Np);

r5full = [r5; -sum(r5)];

R1 = fft2(reshape(r1,Ny,Nx)); R2 = fft2(reshape(r2,Ny,Nx));
R3 = fft2(reshape(r3,Ny,Nx)); R4 = fft2(reshape(r4,Ny,Nx));
R5 = D.liftToFourier(r5full);   % skips liftP's ifft2 + the redundant
                                 % re-transform this used to need

DUf1f = Minv(1,1,:,:).*reshape(R1,1,1,Ny,Nx) + Minv(1,2,:,:).*reshape(R2,1,1,Ny,Nx) ...
      + Minv(1,3,:,:).*reshape(R3,1,1,Ny,Nx) + Minv(1,4,:,:).*reshape(R4,1,1,Ny,Nx) + Minv(1,5,:,:).*reshape(R5,1,1,Ny,Nx);
DUf2f = Minv(2,1,:,:).*reshape(R1,1,1,Ny,Nx) + Minv(2,2,:,:).*reshape(R2,1,1,Ny,Nx) ...
      + Minv(2,3,:,:).*reshape(R3,1,1,Ny,Nx) + Minv(2,4,:,:).*reshape(R4,1,1,Ny,Nx) + Minv(2,5,:,:).*reshape(R5,1,1,Ny,Nx);
DUp1f = Minv(3,1,:,:).*reshape(R1,1,1,Ny,Nx) + Minv(3,2,:,:).*reshape(R2,1,1,Ny,Nx) ...
      + Minv(3,3,:,:).*reshape(R3,1,1,Ny,Nx) + Minv(3,4,:,:).*reshape(R4,1,1,Ny,Nx) + Minv(3,5,:,:).*reshape(R5,1,1,Ny,Nx);
DUp2f = Minv(4,1,:,:).*reshape(R1,1,1,Ny,Nx) + Minv(4,2,:,:).*reshape(R2,1,1,Ny,Nx) ...
      + Minv(4,3,:,:).*reshape(R3,1,1,Ny,Nx) + Minv(4,4,:,:).*reshape(R4,1,1,Ny,Nx) + Minv(4,5,:,:).*reshape(R5,1,1,Ny,Nx);
DPf   = Minv(5,1,:,:).*reshape(R1,1,1,Ny,Nx) + Minv(5,2,:,:).*reshape(R2,1,1,Ny,Nx) ...
      + Minv(5,3,:,:).*reshape(R3,1,1,Ny,Nx) + Minv(5,4,:,:).*reshape(R4,1,1,Ny,Nx) + Minv(5,5,:,:).*reshape(R5,1,1,Ny,Nx);

dUf1 = real(ifft2(reshape(DUf1f,Ny,Nx))); dUf1 = dUf1(:);
dUf2 = real(ifft2(reshape(DUf2f,Ny,Nx))); dUf2 = dUf2(:);
dUp1 = real(ifft2(reshape(DUp1f,Ny,Nx))); dUp1 = dUp1(:);
dUp2 = real(ifft2(reshape(DUp2f,Ny,Nx))); dUp2 = dUp2(:);

% D.truncateFromFourier: matches velocityAv.m's row 5 (TRUNCATION, not an
% ALIASING restrict) -- the preconditioner should restrict pressure's
% correction the same way the true operator does, so it stays a consistent
% approximation of what velocityAv.m actually computes. Also skips a
% redundant fft2 by taking the already-computed DPf directly (previously:
% ifft2(DPf) here, then a fresh fft2 of that result -- an exact round trip
% that cancelled out for nothing).
dPcoarse_full = D.truncateFromFourier(reshape(DPf,Ny,Nx));
dP = dPcoarse_full(1:Np) - dPcoarse_full(NP);

w = [dUf1; dUf2; dUp1; dUp2; dP];

perfCounter('solveVelocityFFT', 'add', 'Pinv', toc(tPinv));
perfCounter('solveVelocityFFT', 'add', 'PinvCalls', 1);
end
