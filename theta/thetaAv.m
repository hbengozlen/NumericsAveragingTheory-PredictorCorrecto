function out = thetaAv(v, ctx)
% Matrix-free operator for solveThetaFFT.m: thetaAv(v) plays the role of
% A*v where A = I/dt + D_adv - (c0/(V0*A_param))*D_diff -- exact, not an
% approximation (same discretization as solveTheta.m's dense D_adv/D_diff,
% just applied without ever assembling a dense matrix). theta lives
% entirely on the fine grid already, so no lift/truncate round trip is
% needed here (unlike phiAv.m) -- just D.DXf/D.DYf/D.D2Xf/D.D2Yf directly.
D = ctx.D; Up1 = ctx.Up1; Up2 = ctx.Up2;
phiext = ctx.phiext; DXphiext = ctx.DXphiext; DYphiext = ctx.DYphiext;
coeff = ctx.coeff; dt_step = ctx.dt_step;

Dv1 = D.DXf(v); Dv2 = D.DYf(v);
D_adv_v  = Up1.*Dv1 + Up2.*Dv2;
D_diff_v = phiext.*(D.D2Xf(v) + D.D2Yf(v)) + DXphiext.*Dv1 + DYphiext.*Dv2;

out = v/dt_step + D_adv_v - coeff*D_diff_v;
end
