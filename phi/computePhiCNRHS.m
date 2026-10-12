function rhs = computePhiCNRHS(phi_old, ctx)
% Explicit RHS for solvePhiCNFFT.m's Crank-Nicolson step:
%   rhs = phi_old/dt - 0.5*L(phi_old),   L(x) = interMat*A_fine*interMatP*x
% Self-contained: same lift/fine-advect/truncate sequence as phiAvCN.m,
% just applied to the KNOWN phi_old instead of the unknown candidate v --
% L itself doesn't care whether its input is the unknown or a known
% vector, only phiAvCN.m's extra v/dt term (absent here, since rhs uses
% phi_old/dt with a MINUS on the 0.5*L(phi_old) term, not a plus) differs.
D = ctx.D; Up1 = ctx.Up1; Up2 = ctx.Up2; divUp = ctx.divUp;
dt_step = ctx.dt_step; Ny = ctx.Ny; Nx = ctx.Nx;

w = D.liftP(phi_old);
fineTerm = Up1.*D.DXf(w) + Up2.*D.DYf(w) + divUp.*w;
Fterm = fft2(reshape(fineTerm, Ny, Nx));
coarseTerm = D.truncateFromFourier(Fterm);

rhs = phi_old/dt_step - 0.5*coarseTerm;
end
