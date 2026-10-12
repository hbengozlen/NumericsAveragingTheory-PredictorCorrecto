function xf = fourierUpsample2D(xc, Nyc, Nxc, Nyf, Nxf)
%FOURIERUPSAMPLE2D Self-contained coarse Nyc x Nxc -> fine Nyf x Nxf
%   spectral lift (exact for content Nyc x Nxc can represent), real space
%   in and out. Generalizes fourierDiffOps.m's liftPImpl -- which
%   special-cases Nyc=Ny-2, Nxc=Nx-2 (this codebase's usual coarse/fine
%   pairing) -- to an ARBITRARY coarse/fine size pair, so callers like
%   plotting_tool.m can upsample straight onto a fixed display grid
%   regardless of the simulation's own grid size. Ax=1 throughout (this
%   codebase's fixed convention -- see fourierGrid(1,N) everywhere).
Ax = 1; L = 2*Ax;
Xc = fft2(reshape(xc, Nyc, Nxc));

%% Pad dim 1 (y): Nyc -> Nyf. Same two corrections as liftPImpl: (1) the
% ambiguous coarse Nyquist bin is SPLIT in half between the fine grid's
% +Nyquist/-Nyquist bins; (2) a per-mode linear phase correction, since
% fourierGrid(1,Nc) and fourierGrid(1,Nf) place their first grid point at
% slightly different physical x. Skipped entirely if this axis needs no
% padding (Nyc==Nyf would otherwise double-write the shared Nyquist index).
if Nyc == Nyf
    tmp = Xc;
else
    half1 = Nyc/2;
    shift1 = 2*Ax*(1/Nyf - 1/Nyc);
    tmp = zeros(Nyf, Nxc);
    k_pos1 = (0:half1-1)';
    tmp(1:half1,:) = Xc(1:half1,:) .* exp(1i*2*pi*k_pos1*shift1/L);
    phase_nyq1 = exp(1i*2*pi*half1*shift1/L);
    tmp(half1+1,:)         = Xc(half1+1,:)/2 * phase_nyq1;
    tmp(Nyf-half1+1,:)     = tmp(Nyf-half1+1,:) + Xc(half1+1,:)/2 * conj(phase_nyq1);
    k_neg1 = (-(half1-1):-1)';
    tmp(Nyf-half1+2:Nyf,:) = Xc(half1+2:Nyc,:) .* exp(1i*2*pi*k_neg1*shift1/L);
    tmp = tmp * (Nyf/Nyc);
end

%% Pad dim 2 (x): Nxc -> Nxf, via transpose (same math, other axis).
tmpT = tmp.';
if Nxc == Nxf
    XfT = tmpT;
else
    half2 = Nxc/2;
    shift2 = 2*Ax*(1/Nxf - 1/Nxc);
    XfT = zeros(Nxf, Nyf);
    k_pos2 = (0:half2-1)';
    XfT(1:half2,:) = tmpT(1:half2,:) .* exp(1i*2*pi*k_pos2*shift2/L);
    phase_nyq2 = exp(1i*2*pi*half2*shift2/L);
    XfT(half2+1,:)         = tmpT(half2+1,:)/2 * phase_nyq2;
    XfT(Nxf-half2+1,:)     = XfT(Nxf-half2+1,:) + tmpT(half2+1,:)/2 * conj(phase_nyq2);
    k_neg2 = (-(half2-1):-1)';
    XfT(Nxf-half2+2:Nxf,:) = tmpT(half2+2:Nxc,:) .* exp(1i*2*pi*k_neg2*shift2/L);
    XfT = XfT * (Nxf/Nxc);
end
Xf = XfT.';

xf = real(ifft2(Xf));
xf = xf(:);
end
