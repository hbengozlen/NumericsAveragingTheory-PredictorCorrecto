function theta_new = solveTheta(theta_old, phi, Uf1, Uf2, Up1, Up2, dt_step, parameter, CoarseMesh)
% Implicit Backward Euler step for angular momentum:
%   (theta_new - theta_old)/dt = L_theta*theta_new + f_src
% where L_theta contains diffusion (c0*div(phi*grad)) and advection (-Up*grad).
% f_src = 1/2*(dy*Up1 - dx*Up2) - ksif/2*(dy*Uf1 - dx*Uf2)  (explicit vorticity source)
% All quantities on fine grid; phi interpolated from coarse via interMatP.

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

% Diffusion operator on fine grid (linear in theta): div(phi*grad(theta)) =
% phi*Laplacian(theta) + grad(phi).grad(theta), i.e. the product-rule
% expansion, not the conservative DX*diag(phi)*DX form -- that form applies
% DX to theta FIRST, which zeros theta's own Nyquist mode before phi ever
% gets multiplied in (DX*diag(phi)*DX*nyquist_mode = 0 regardless of phi),
% same root cause as solveVelocity.m's viscosity terms. D2X/D2Y (direct
% 2nd-derivative matrices) don't have that blind spot.
D_diff = diag(phiext)*(D2X + D2Y) ...
       + diag(DX*phiext)*DX + diag(DY*phiext)*DY;

% Advection operator (linear in theta)
D_adv = diag(Up1)*DX + diag(Up2)*DY;

% Explicit vorticity source (no theta)
f_src = 0.5*(DY*Up1 - DX*Up2) - (ksif/2)*(DY*Uf1 - DX*Uf2);

% BE: (I/dt - L_theta) * theta_new = theta_old/dt + f_src
A   = eye(Nfine)/dt_step + D_adv - (c0/(V0*A_param)) * D_diff ;
rhs = theta_old/dt_step + f_src;

theta_new = A \ rhs;
end
