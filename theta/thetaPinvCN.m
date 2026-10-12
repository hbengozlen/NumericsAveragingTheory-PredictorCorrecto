function w = thetaPinvCN(r, ctx)
% Applies solveThetaCNFFT.m's frozen-phibar/frozen-Up preconditioner --
% same lift-free divide sequence as thetaPinv.m, but divides by the CN
% denominator (1/dt - 0.5*(coeff*phibar*(KX2+KY2) - a*KX - b*KY))
% precomputed in ctx.invDenomCN, matching thetaAvCN.m's (I/dt - 0.5*
% L_theta) instead of thetaAv.m's (I/dt + D_adv - coeff*D_diff).
Fr = fft2(reshape(r, ctx.Ny, ctx.Nx));
Fcorrected = Fr .* ctx.invDenomCN;
w = real(ifft2(Fcorrected));
w = w(:);
end
