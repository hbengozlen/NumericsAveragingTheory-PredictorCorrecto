function [Uf1, Uf2, Up1, Up2, P] = solveVelocity(phi, theta, parameter, FineMesh, CoarseMesh)
% Solves the coupled velocity-pressure system given frozen phi and theta.
% No inertia (Stokes limit): fluid mom X/Y, particle mom X/Y, fluid continuity.
% Unknowns: [Uf1; Uf2; Up1; Up2; P]
%   Uf1,Uf2,Up1,Up2 on fine grid (Nfine each)
%   P on reduced coarse grid (Ncoarse-1, pressure pinning removes one DOF)

tBuild = tic;

DX         = parameter.DX;
DY         = parameter.DY;
A_param    = parameter.A;
etaf       = parameter.etaf;
etab       = parameter.etab;
R          = parameter.R;
b          = parameter.b;
V0         = parameter.V0;
Db         = parameter.Db;
PrefPt     = parameter.PrefPt;
Nx         = parameter.Nx;
Ny         = parameter.Ny;
boolactive = parameter.boolactive;
boolpusher = parameter.boolpusher;
alpha      = -2*boolpusher + 1;

interMatP = CoarseMesh.interMatP;   % coarse -> fine
interMatU  = FineMesh.interMatU;     % fine   -> coarse

Ncoarse = (Nx-2)*(Ny-2);
Nfine   = Nx*Ny;
Np      = Ncoarse - 1;

%% Pressure pinning: mean(P)=PrefPt, via the cheap (non-orthonormal) basis
% v_i = e_i - e_Ncoarse for i=1..Np -- spans the same zero-mean subspace as
% null(ones(1,Ncoarse)) (every column still sums to 0), but is O(Np) to
% write down instead of an SVD, and its action is a plain concatenation
% instead of a dense matrix-vector product (see solveVelocityFFT.m's Av/
% Pinv, which use it as [dP;-sum(dP)] / v(1:Np)-v(end) directly instead of
% ever forming this matrix). Formed explicitly here only because this
% solver assembles NmatP into other dense matrices (GX, B51..B54) below.
NmatP = [eye(Np); -ones(1,Np)];      % Ncoarse x Np, every column zero-mean
Ptilde = PrefPt * ones(Ncoarse,1);   % mean(Ptilde) = PrefPt exactly

%% Interpolated quantities on fine grid
phiext = interMatP * phi;
d1 = cos(theta);
d2 = sin(theta);

ksi_A2 = 2*A_param^2/R^2 * phiext;       % ksi*(A^2/etaf) = 2*etaf/R^2*phi * A^2/etaf
ksi_Ab = 2*A_param*b/R^2 * phiext;       % ksi*(A*b/etaf)

Xi   = diag(ksi_A2);
phi1 = diag(1 - phiext);
phid = diag(phiext);

%% Differential operator products
% DXX,DYY use the DIRECT second-derivative matrix (fourierGrid2.m), not
% DX*DX/DY*DY. DX zeros the Nyquist mode's derivative (its direction is
% ambiguous on an even grid), and DX*DX inherits that zero, which is what
% made the viscosity terms blind to that wavenumber and forced the
% Mscale*Cexcess penalty below. A direct second derivative has no such
% ambiguity (curvature is well-defined regardless of the Nyquist mode's
% assumed sign) and gives the correct nonzero value there instead --
% closing that null space at the source. Verified: cond(A_sys) drops from
% ~1e17 to ~1.8e6 with this swap alone, no penalty term needed.
[~, D2x] = fourierGrid2(1, Nx);
[~, D2y] = fourierGrid2(1, Ny);
DXX = kron(D2x, eye(Ny));  DYY = kron(eye(Nx), D2y);
DXY = DX*DY;  DYX = DY*DX;   % mixed partial: still ambiguous at the Nyquist
                             % mode in either axis, no direct-formula fix exists

%% Pressure gradient operators: fine x reduced-coarse
GX = DX * interMatP * NmatP;    % Nfine x Np
GY = DY * interMatP * NmatP;

%% Block matrix assembly
Z = zeros(Nfine, Nfine);

% Row 1: fluid momentum X  (-ksi(Uf1-Up1) - (1-phi)*grad_x P + visc_x = 0)
B11 = 2*DXX + DYY - Xi;
B12 = DYX;
B13 = Xi;
B14 = Z;
B15 = -phi1 * GX;

% Row 2: fluid momentum Y  (-ksi(Uf2-Up2) - (1-phi)*grad_y P + visc_y = 0)
B21 = DXY;
B22 = DXX + 2*DYY - Xi;
B23 = Z;
B24 = Xi;
B25 = -phi1 * GY;

% Row 3: particle momentum X  (ksi(Uf1-Up1) - phi*grad_x P + visc_x_p = 0)
B31 = Xi;
B32 = Z;
B33 = (etab/etaf)*(2*DXX + DYY) - Xi;
B34 = (etab/etaf)*DYX;
B35 = -phid * GX;

% Row 4: particle momentum Y  (ksi(Uf2-Up2) - phi*grad_y P + visc_y_p = 0)
B41 = Z;
B42 = Xi;
B43 = (etab/etaf)*DXY;
B44 = (etab/etaf)*(DXX + 2*DYY) - Xi;
B45 = -phid * GY;

% Row 5: fluid continuity  (NmatP'*interMat*div((1-phi)Uf + phi*Up) = 0)
B51 = NmatP' * interMatU * DX * phi1;   % Np x Nfine
B52 = NmatP' * interMatU * DY * phi1;
B53 = NmatP' * interMatU * DX * phid;
B54 = NmatP' * interMatU * DY * phid;
B55 = zeros(Np, Np);

%% Row scaling
% The continuity row (B51..B55) has no drag/viscous prefactor like the momentum
% rows (which are dominated by Xi ~ 2*A^2/R^2*phi and the O(N^2) viscous terms),
% so it comes out orders of magnitude smaller and skews the conditioning of A_sys.
% Rescale it to match the momentum rows' norm; this leaves the solution unchanged
% since it's just multiplying one block-row of equations by a nonzero constant.
rowScale = norm([B11 B12 B13 B14 B15], 'fro') / norm([B51 B52 B53 B54 B55], 'fro');
B51 = rowScale*B51;  B52 = rowScale*B52;
B53 = rowScale*B53;  B54 = rowScale*B54;  B55 = rowScale*B55;

%% RHS vectors  (explicit terms moved to right-hand side)
Ptilde_fine = interMatP * Ptilde;
DxP0 = DX * Ptilde_fine;
DyP0 = DY * Ptilde_fine;

if boolactive
    f1 = ksi_A2.*d1 + (1-phiext).*DxP0 ...
       - DX*(alpha*ksi_Ab.*d1.*d1) - DY*(alpha*ksi_Ab.*d1.*d2);
    f2 = ksi_A2.*d2 + (1-phiext).*DyP0 ...
       - DX*(alpha*ksi_Ab.*d2.*d1) - DY*(alpha*ksi_Ab.*d2.*d2);
    f3 = -ksi_A2.*d1 + phiext.*DxP0 + (2*A_param/R^2/V0)*Db*(DX*phiext);
    f4 = -ksi_A2.*d2 + phiext.*DyP0 + (2*A_param/R^2/V0)*Db*(DY*phiext);
else
    f1 = (1-phiext).*DxP0;
    f2 = (1-phiext).*DyP0;
    f3 = phiext.*DxP0 + (2*A_param/R^2/V0)*Db*(DX*phiext);
    f4 = phiext.*DyP0 + (2*A_param/R^2/V0)*Db*(DY*phiext);
end
f5 = rowScale * zeros(Np, 1);

%% Velocity gauge pinning
% The momentum equations only ever involve Uf-Up differences and spatial
% derivatives, so shifting Uf1 and Up1 together by the same constant (or Uf2,Up2
% together) leaves every residual unchanged -- A_sys has an exact 2D null space
% (uniform drift in x and in y). Fix it the same way P's constant-shift freedom
% is fixed: replace the now-redundant row at one reference index with a gauge
% constraint. Uses mean(Uf1)=0 / mean(Uf2)=0, not a physical point value --
% fft2(x)(1,1)=sum(x), so this is EXACTLY the DC-mode Fourier coefficient of
% Uf1/Uf2, a per-mode constraint the FFT-based preconditioner in
% solveVelocityFFT.m can represent exactly (a physical point-pin spreads
% across every Fourier mode when transformed; this doesn't) -- keeping this
% dense solver's gauge choice consistent with that one.
RefPtIDXfine = Ny*(Nx/2 - 1) + Ny/2;

B11(RefPtIDXfine,:) = ones(1,Nfine)/Nfine;  B12(RefPtIDXfine,:) = 0;  B13(RefPtIDXfine,:) = 0;
B14(RefPtIDXfine,:) = 0;  B15(RefPtIDXfine,:) = 0;
f1(RefPtIDXfine) = 0;                     % pins mean(Uf1) = 0

B21(RefPtIDXfine,:) = 0;  B22(RefPtIDXfine,:) = ones(1,Nfine)/Nfine;  B23(RefPtIDXfine,:) = 0;
B24(RefPtIDXfine,:) = 0;  B25(RefPtIDXfine,:) = 0;
f2(RefPtIDXfine) = 0;                     % pins mean(Uf2) = 0

%% Full system and solve
A_sys = [B11  B12  B13  B14  B15;
         B21  B22  B23  B24  B25;
         B31  B32  B33  B34  B35;
         B41  B42  B43  B44  B45;
         B51  B52  B53  B54  B55];

rhs = [f1; f2; f3; f4; f5];

perfCounter('solveVelocity', 'add', 'build', toc(tBuild));

tSolve = tic;
sol  = A_sys \ rhs;
perfCounter('solveVelocity', 'add', 'solve', toc(tSolve));
perfCounter('solveVelocity', 'add', 'calls', 1);

Uf1 = sol(1           : Nfine);
Uf2 = sol(Nfine+1     : 2*Nfine);
Up1 = sol(2*Nfine+1   : 3*Nfine);
Up2 = sol(3*Nfine+1   : 4*Nfine);
P   = sol(4*Nfine+1   : 4*Nfine+Np);
end
