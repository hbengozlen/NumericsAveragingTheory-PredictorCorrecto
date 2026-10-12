function out = phiAvCN(v, ctx)
% Matrix-free operator for solvePhiCNFFT.m: phiAvCN(v) plays the role of
% A*v where A = interMat*(I/dt + 0.5*A_fine)*interMatP, A_fine = Up.grad(.)
% + div(Up)*(.) -- exact, not an approximation (same discretization as
% solvePhiCN.m's dense A_fine, just applied without ever assembling a
% dense matrix). Same lift/fine-advect/truncate sequence as phiAv.m (see
% that file), just with the extra 0.5 factor Crank-Nicolson puts on the
% implicit operator -- self-contained rather than sharing code with
% phiAv.m, matching this codebase's convention of one function showing its
% whole computation (see fourierDiffOps.m's header for the same rationale).
D = ctx.D; Up1 = ctx.Up1; Up2 = ctx.Up2; divUp = ctx.divUp;
dt_step = ctx.dt_step; Ny = ctx.Ny; Nx = ctx.Nx;

w = D.liftP(v);
fineTerm = Up1.*D.DXf(w) + Up2.*D.DYf(w) + divUp.*w;
Fterm = fft2(reshape(fineTerm, Ny, Nx));
coarseTerm = D.truncateFromFourier(Fterm);

out = v/dt_step + 0.5*coarseTerm;
end
