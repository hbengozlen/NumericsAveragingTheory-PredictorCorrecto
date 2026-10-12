function w = thetaPinv(r, ctx)
% Applies solveThetaFFT.m's frozen-phibar/frozen-Up preconditioner:
% freezing phiext to its mean phibar (so grad(phibar)=0, dropping
% D_diff's DXphiext/DYphiext terms entirely) and Up1,Up2 to their means
% makes the operator constant-coefficient, hence diagonal per Fourier
% mode -- divide r's own Fourier transform by the per-mode scalar
% (1/dt + a*KX+b*KY - coeff*phibar*(KX2+KY2)) precomputed in
% ctx.invDenom. No lift/truncate needed: theta lives entirely on the fine
% grid already, so this is a plain fft2/ifft2, not D.liftToFourier/
% D.truncateFromFourier.
Fr = fft2(reshape(r, ctx.Ny, ctx.Nx));
Fcorrected = Fr .* ctx.invDenom;
w = real(ifft2(Fcorrected));
w = w(:);
end
