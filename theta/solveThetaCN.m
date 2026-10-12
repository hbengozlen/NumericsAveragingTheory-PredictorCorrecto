function theta_new = solveThetaCN(theta_old, phi, Uf1, Uf2, Up1, Up2, dt_step, parameter, CoarseMesh)
% Crank-Nicolson step for angular momentum:
%   (theta_new - theta_old)/dt = 1/2*L_theta*theta_new + 1/2*L_theta*theta_old + f_src
% Rearranged: (I/dt - 1/2*L_theta)*theta_new = (I/dt + 1/2*L_theta)*theta_old + f_src
% phi and u are the half-step values (frozen coefficients).

DX       = parameter.DX;
DY       = parameter.DY;
D2X      = parameter.D2X;
D2Y      = parameter.D2Y;
A_param  = parameter.A;
V0       = parameter.V0;
etaf     = parameter.etaf;
R        = parameter.R;
L_param  = parameter.L;
elasPara = parameter.elasPara;
ksif     = parameter.ksif;
interMatP = CoarseMesh.interMatP;
Nfine    = length(theta_old);

c0 = elasPara * 6*R^2 / (etaf * L_param^2);

phiext = interMatP * phi;

% See solveTheta.m for why D2X/D2Y (direct 2nd-derivative) replace the
% conservative DX*diag(phi)*DX form here.
D_diff = diag(phiext)*(D2X + D2Y) ...
       + diag(DX*phiext)*DX + diag(DY*phiext)*DY;

D_adv = -diag(Up1)*DX - diag(Up2)*DY;

L_theta = (c0/(V0*A_param)) * D_diff + D_adv;

f_src = 0.5*(DY*Up1 - DX*Up2) - (ksif/2)*(DY*Uf1 - DX*Uf2);

A   = eye(Nfine)/dt_step - 0.5*L_theta;
rhs = (eye(Nfine)/dt_step + 0.5*L_theta)*theta_old + f_src;

theta_new = A \ rhs;
end
