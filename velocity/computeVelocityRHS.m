function rhs = computeVelocityRHS(phiext, ksi_A2, ksi_Ab, d1, d2, alpha, boolactive, ...
    A_param, R, V0, Db, Ptilde, Np, RefPtIDXfine, D)
% RHS vectors (explicit terms) for solveVelocityFFT.m, mirrors
% solveVelocity.m exactly via D.DXf/D.DYf/D.liftP.
Ptilde_fine = D.liftP(Ptilde);
DxP0 = D.DXf(Ptilde_fine);
DyP0 = D.DYf(Ptilde_fine);

if boolactive
    f1 = ksi_A2.*d1 + (1-phiext).*DxP0 ...
       - D.DXf(alpha*ksi_Ab.*d1.*d1) - D.DYf(alpha*ksi_Ab.*d1.*d2);
    f2 = ksi_A2.*d2 + (1-phiext).*DyP0 ...
       - D.DXf(alpha*ksi_Ab.*d2.*d1) - D.DYf(alpha*ksi_Ab.*d2.*d2);
    f3 = -ksi_A2.*d1 + phiext.*DxP0 + (2*A_param/R^2/V0)*Db*(D.DXf(phiext));
    f4 = -ksi_A2.*d2 + phiext.*DyP0 + (2*A_param/R^2/V0)*Db*(D.DYf(phiext));
else
    f1 = (1-phiext).*DxP0;
    f2 = (1-phiext).*DyP0;
    f3 = phiext.*DxP0 + (2*A_param/R^2/V0)*Db*(D.DXf(phiext));
    f4 = phiext.*DyP0 + (2*A_param/R^2/V0)*Db*(D.DYf(phiext));
end
f5 = zeros(Np,1);

f1(RefPtIDXfine) = 0;   % gauge pin RHS (matches velocityAv's pin)
f2(RefPtIDXfine) = 0;

rhs = [f1; f2; f3; f4; f5];
end
