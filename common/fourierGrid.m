function [x,Dx]=fourierGrid(Ax,Nx)
    dx = 2*Ax/Nx; x = -Ax+dx*(1:Nx)'; hx=2*pi/Nx;
    column = [0 .5*(-1).^(1:Nx-1).*cot((1:Nx-1)*hx/2)]';
    Dx = pi/Ax*toeplitz(column,column([1 Nx:-1:2]));
end
