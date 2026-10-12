function rhs = computeThetaRHS(theta_old, Uf1, Uf2, Up1, Up2, dt_step, ksif, D)
% Explicit RHS for solveThetaFFT.m: rhs = theta_old/dt + f_src, f_src the
% explicit vorticity source (no theta dependence) -- matches solveTheta.m's
% f_src exactly, via D.DXf/D.DYf instead of dense DX/DY.
f_src = 0.5*(D.DYf(Up1) - D.DXf(Up2)) - (ksif/2)*(D.DYf(Uf1) - D.DXf(Uf2));
rhs = theta_old/dt_step + f_src;
end
