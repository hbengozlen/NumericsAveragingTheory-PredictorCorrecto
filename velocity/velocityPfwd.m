function w = velocityPfwd(r, ctx)
% Applies the frozen-phibar preconditioner's FORWARD block M (ctx.M, from
% buildVelocityPreconditioner.m's second output) -- NOT its inverse Minv.
% Used only once per solveVelocityFFT.m call, to build a warm-start
% initial guess for GMRES: under right preconditioning GMRES solves for y
% where x=velocityPinv(y), so a PHYSICAL initial guess x0 must be mapped
% through this forward block first (y0=velocityPfwd(x0,ctx)) before it
% means anything as an initial guess for y -- if x0 were exactly the
% frozen system's fixed point, velocityPinv(y0) would reproduce it
% exactly, giving GMRES a much better starting point than the zero vector.
% Same FFT/per-mode structure as velocityPinv.m, just M instead of Minv
% (and no perfCounter bookkeeping -- this isn't inside GMRES's iteration).
Nx = ctx.Nx; Ny = ctx.Ny; Nfine = ctx.Nfine; Np = ctx.Np; NP = ctx.NP;
D = ctx.D; M = ctx.M;

r1 = r(1:Nfine); r2 = r(Nfine+1:2*Nfine);
r3 = r(2*Nfine+1:3*Nfine); r4 = r(3*Nfine+1:4*Nfine);
r5 = r(4*Nfine+1:4*Nfine+Np);

r5full = [r5; -sum(r5)];

R1 = fft2(reshape(r1,Ny,Nx)); R2 = fft2(reshape(r2,Ny,Nx));
R3 = fft2(reshape(r3,Ny,Nx)); R4 = fft2(reshape(r4,Ny,Nx));
R5 = D.liftToFourier(r5full);

DUf1f = M(1,1,:,:).*reshape(R1,1,1,Ny,Nx) + M(1,2,:,:).*reshape(R2,1,1,Ny,Nx) ...
      + M(1,3,:,:).*reshape(R3,1,1,Ny,Nx) + M(1,4,:,:).*reshape(R4,1,1,Ny,Nx) + M(1,5,:,:).*reshape(R5,1,1,Ny,Nx);
DUf2f = M(2,1,:,:).*reshape(R1,1,1,Ny,Nx) + M(2,2,:,:).*reshape(R2,1,1,Ny,Nx) ...
      + M(2,3,:,:).*reshape(R3,1,1,Ny,Nx) + M(2,4,:,:).*reshape(R4,1,1,Ny,Nx) + M(2,5,:,:).*reshape(R5,1,1,Ny,Nx);
DUp1f = M(3,1,:,:).*reshape(R1,1,1,Ny,Nx) + M(3,2,:,:).*reshape(R2,1,1,Ny,Nx) ...
      + M(3,3,:,:).*reshape(R3,1,1,Ny,Nx) + M(3,4,:,:).*reshape(R4,1,1,Ny,Nx) + M(3,5,:,:).*reshape(R5,1,1,Ny,Nx);
DUp2f = M(4,1,:,:).*reshape(R1,1,1,Ny,Nx) + M(4,2,:,:).*reshape(R2,1,1,Ny,Nx) ...
      + M(4,3,:,:).*reshape(R3,1,1,Ny,Nx) + M(4,4,:,:).*reshape(R4,1,1,Ny,Nx) + M(4,5,:,:).*reshape(R5,1,1,Ny,Nx);
DPf   = M(5,1,:,:).*reshape(R1,1,1,Ny,Nx) + M(5,2,:,:).*reshape(R2,1,1,Ny,Nx) ...
      + M(5,3,:,:).*reshape(R3,1,1,Ny,Nx) + M(5,4,:,:).*reshape(R4,1,1,Ny,Nx) + M(5,5,:,:).*reshape(R5,1,1,Ny,Nx);

dUf1 = real(ifft2(reshape(DUf1f,Ny,Nx))); dUf1 = dUf1(:);
dUf2 = real(ifft2(reshape(DUf2f,Ny,Nx))); dUf2 = dUf2(:);
dUp1 = real(ifft2(reshape(DUp1f,Ny,Nx))); dUp1 = dUp1(:);
dUp2 = real(ifft2(reshape(DUp2f,Ny,Nx))); dUp2 = dUp2(:);

dPcoarse_full = D.truncateFromFourier(reshape(DPf,Ny,Nx));
dP = dPcoarse_full(1:Np) - dPcoarse_full(NP);

w = [dUf1; dUf2; dUp1; dUp2; dP];
end
