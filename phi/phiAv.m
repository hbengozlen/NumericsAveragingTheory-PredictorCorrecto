function out = phiAv(v, ctx)
% Matrix-free operator for solvePhiFFT.m: phiAv(v) plays the role of A*v
% where A = interMat*(I/dt + A_fine)*interMatP, A_fine = Up.grad(.) +
% div(Up)*(.) -- exact, not an approximation (same discretization as
% solvePhi.m's dense A_fine, just applied without ever assembling a dense
% matrix). Uses D.liftP (coarse->fine, real space) and D.DXf/D.DYf
% (FFT-based fine-grid derivatives) to build the fine-grid action, then
% D.truncateFromFourier (fine->coarse, spectral TRUNCATION -- matches
% FineMesh.interMatU = buildRestrictTruncMat, NOT an aliasing restrict) to
% restrict back to the coarse grid v itself lives on.
D = ctx.D; Up1 = ctx.Up1; Up2 = ctx.Up2; divUp = ctx.divUp;
dt_step = ctx.dt_step; Ny = ctx.Ny; Nx = ctx.Nx;

w = D.liftP(v);
fineTerm = Up1.*D.DXf(w) + Up2.*D.DYf(w) + divUp.*w;
Fterm = fft2(reshape(fineTerm, Ny, Nx));
coarseTerm = D.truncateFromFourier(Fterm);

out = v/dt_step + coarseTerm;
end
