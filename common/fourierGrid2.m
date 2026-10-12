function [x,D2]=fourierGrid2(Ax,Nx)
% Direct second-order spectral differentiation matrix (Trefethen, "Spectral
% Methods in MATLAB", eq. 3.11-3.12) -- NOT the same as fourierGrid's Dx
% squared (Dx*Dx). Dx zeros the Nyquist mode's derivative (its direction is
% ambiguous for a real signal on an even grid), and that zero propagates
% through when you square it. The direct second-derivative formula has no
% such ambiguity -- a curvature is well-defined regardless of which sign
% convention you'd assign the Nyquist wavenumber -- so D2's Nyquist-mode
% eigenvalue is the correct nonzero value, unlike Dx*Dx's.
%
% Same (x, Ax, Nx) conventions as fourierGrid.m: domain [-Ax,Ax), Nx points,
% Nx even.
    dx = 2*Ax/Nx; x = -Ax+dx*(1:Nx)'; hx=2*pi/Nx;
    j = (1:Nx-1)';
    column = [-pi^2/(3*hx^2) - 1/6; -0.5*(-1).^j./sin(j*hx/2).^2];
    D2 = (pi/Ax)^2 * toeplitz(column);
end
