function phi_new = solvePhi(phi_old, Up1, Up2, dt_step, parameter, FineMesh, CoarseMesh)
% Implicit Backward Euler step for particle continuity:
%   (phi_new - phi_old)/dt + div(phi_new * Up) = 0
% phi lives on coarse grid; computation projected via interMatP/interMat.

DX        = parameter.DX;
DY        = parameter.DY;
interMatP = CoarseMesh.interMatP;   % coarse -> fine
interMat  = FineMesh.interMatU;     % fine   -> coarse
Nfine     = length(Up1);

divUp  = DX*Up1 + DY*Up2;
A_fine = diag(Up1)*DX + diag(Up2)*DY + diag(divUp);

A   = interMat * (eye(Nfine)/dt_step + A_fine) * interMatP;
rhs = interMat * interMatP * phi_old / dt_step;

phi_new = A \ rhs;
end
