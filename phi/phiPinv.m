function w = phiPinv(r, ctx)
% Applies solvePhiFFT.m's frozen-Up preconditioner: freezing Up1,Up2 to
% their spatial means (and div(Up) to ITS mean) makes phiAv's fine-grid
% operator constant-coefficient, hence diagonal per Fourier mode -- lift r
% to the fine grid's OWN Fourier transform (D.liftToFourier), divide by
% the per-mode scalar (1/dt + a*KX + b*KY + c) precomputed in
% ctx.invDenom, then truncate back to the coarse grid
% (D.truncateFromFourier) -- the SAME lift/truncate sequence phiAv itself
% uses, so this is an exact frozen-coefficient approximation of phiAv, not
% a different operator applied to a different space.
D = ctx.D;
Fr = D.liftToFourier(r);
Fcorrected = Fr .* ctx.invDenom;
w = D.truncateFromFourier(Fcorrected);
end
