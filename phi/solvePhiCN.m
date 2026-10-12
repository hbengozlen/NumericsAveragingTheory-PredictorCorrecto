function phi_new = solvePhiCN(phi_old, Up1, Up2, dt_step, parameter, FineMesh, CoarseMesh)
% Crank-Nicolson step for particle continuity:
%   (phi_new - phi_old)/dt + 1/2*div(phi_new*Up) + 1/2*div(phi_old*Up) = 0
% phi lives on coarse grid; computation projected via interMatP/interMat.

DX        = parameter.DX;
DY        = parameter.DY;
interMatP = CoarseMesh.interMatP;
interMat  = FineMesh.interMatU;
Nfine     = length(Up1);

divUp  = DX*Up1 + DY*Up2;
A_fine = diag(Up1)*DX + diag(Up2)*DY + diag(divUp);

A   = interMat * (eye(Nfine)/dt_step + 0.5*A_fine) * interMatP;
rhs = interMat * (eye(Nfine)/dt_step - 0.5*A_fine) * interMatP * phi_old;

phi_new = A \ rhs;
end
