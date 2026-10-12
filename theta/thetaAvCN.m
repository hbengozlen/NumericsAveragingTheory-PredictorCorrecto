function out = thetaAvCN(v, ctx)
% Matrix-free operator for solveThetaCNFFT.m: thetaAvCN(v) plays the role
% of A*v where A = I/dt - 0.5*L_theta, L_theta = coeff*D_diff + D_adv,
% D_adv = -Up.grad(.) -- literal transcription of solveThetaCN.m's dense
% D_diff/D_adv/L_theta/A (same sign convention, including D_adv's minus
% sign -- see solveThetaCN.m's comment for why), just applied without
% ever assembling a dense matrix. theta lives entirely on the fine grid
% already, so no lift/truncate round trip is needed (unlike phiAvCN.m).
D = ctx.D; Up1 = ctx.Up1; Up2 = ctx.Up2;
phiext = ctx.phiext; DXphiext = ctx.DXphiext; DYphiext = ctx.DYphiext;
coeff = ctx.coeff; dt_step = ctx.dt_step;

Dv1 = D.DXf(v); Dv2 = D.DYf(v);
D_diff_v = phiext.*(D.D2Xf(v) + D.D2Yf(v)) + DXphiext.*Dv1 + DYphiext.*Dv2;
D_adv_v  = -Up1.*Dv1 - Up2.*Dv2;
L_theta_v = coeff*D_diff_v + D_adv_v;

out = v/dt_step - 0.5*L_theta_v;
end
