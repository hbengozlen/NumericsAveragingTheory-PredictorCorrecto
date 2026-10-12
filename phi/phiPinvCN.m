function w = phiPinvCN(r, ctx)
% Applies solvePhiCNFFT.m's frozen-Up preconditioner -- same lift/divide/
% truncate sequence as phiPinv.m, but divides by the CN denominator
% (1/dt + 0.5*(a*KX+b*KY+c)) precomputed in ctx.invDenomCN, matching
% phiAvCN.m's (I/dt + 0.5*Lfrozen) instead of phiAv.m's (I/dt+Lfrozen).
D = ctx.D;
Fr = D.liftToFourier(r);
Fcorrected = Fr .* ctx.invDenomCN;
w = D.truncateFromFourier(Fcorrected);
end
