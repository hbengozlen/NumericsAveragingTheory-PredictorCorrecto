function [Minv, M] = buildVelocityPreconditioner(phibar, parameter, D)
%BUILDVELOCITYPRECONDITIONER Frozen-phibar, per-Fourier-mode 5x5 block
%   preconditioner for solveVelocityFFT.m's saddle-point system
%   [Uf1;Uf2;Up1;Up2;P]. Freezing phi to its spatial mean makes the
%   operator exactly block-diagonal per shared wavenumber (kx,ky) -- one
%   small 5x5 solve per mode instead of one huge dense factorization. D is
%   a fourierDiffOps(Nx,Ny) struct (uses D.KX/D.KY/D.KX2/D.KY2 directly).
%
%   Second output M (optional): the forward, UNINVERTED per-mode block,
%   same Ny x Nx x 5 x 5 layout as Minv. Callers that only need to apply
%   the preconditioner (velocityPinv.m) only need Minv; M itself is
%   exposed for velocityPfwd.m, which warm-starts solveVelocityFFT.m's
%   GMRES by mapping a physical initial guess through this forward block
%   (the operation Minv itself inverts).
%
%   Two explicit decouplings make M full rank at every single mode, so
%   plain inv() (not pinv) is safe throughout:
%
%   1) At the true DC mode (kx=ky=0), rows 1,2 (fluid momentum) are
%      replaced by a literal identity -- the exact Fourier-space
%      representation of solveVelocityFFT.m's Av mean(Uf1)=0/mean(Uf2)=0
%      gauge pin (fft2(x)(1,1)=sum(x)), not an approximation.
%
%   2) Row/col 5 (pressure) is decoupled (set to a trivial identity)
%      wherever pressure has no genuine degree of freedom there. Two
%      alternative extents for this are kept in the code below (one
%      commented out) -- currently ACTIVE: Option B, decouple every fine
%      mode with NO coarse-grid pressure counterpart, an always-3-wide gap
%      band per axis (see Option B's comment below for why it's 3 wide,
%      and the known extra approximation in its 2 flanking bins); this
%      does NOT cover the DC mode on its own, so decoupling (2a) below
%      handles that explicitly. The narrower alternative (Option A,
%      commented out): decouple ONLY the 4 corner modes where kx,ky in
%      {0,Nyquist} simultaneously. Option A is FULL RANK everywhere but
%      NOT well-conditioned everywhere -- at any non-corner mode with
%      exactly one of kx,ky at Nyquist (where that axis's first-derivative
%      eigenvalue is deliberately zeroed) and the other axis at a small
%      wavenumber, pressure's only remaining coupling is through that
%      small eigenvalue, giving a tiny effective pivot for P: cond(M) was
%      measured up to ~1.5e6 there (vs ~1.1e5 under Option B), and
%      inv(M)'s resulting huge entries amplify whatever residual GMRES
%      puts on those modes -- verified to blow up the solution (10-50x
%      relative error, GMRES stagnation) once phi/theta forcing has enough
%      spectral content to excite them, e.g. at larger theta amplitude.
%      Prefer Option B; keep Option A only for deliberately re-testing
%      this failure mode. Swap which block is commented out to switch.

Nx = parameter.Nx; Ny = parameter.Ny;
A_param = parameter.A; R = parameter.R;
etab = parameter.etab; etaf = parameter.etaf;

KX = D.KX; KY = D.KY; KX2 = D.KX2; KY2 = D.KY2;
kappaPhibar = 2*A_param^2/R^2*phibar;

M = zeros(5,5,Ny,Nx);
M(1,1,:,:) = -kappaPhibar + 2*KX2+KY2;   M(1,2,:,:) = KX.*KY;
M(1,3,:,:) = kappaPhibar;                    M(1,5,:,:) = -(1-phibar)*KX;
M(2,1,:,:) = KX.*KY;                         M(2,2,:,:) = -kappaPhibar + KX2+2*KY2;
M(2,4,:,:) = kappaPhibar;                    M(2,5,:,:) = -(1-phibar)*KY;
M(3,1,:,:) = kappaPhibar;
M(3,3,:,:) = -kappaPhibar + (etab/etaf)*(2*KX2+KY2); M(3,4,:,:) = (etab/etaf)*KX.*KY;
M(3,5,:,:) = -phibar*KX;
M(4,2,:,:) = kappaPhibar;                    M(4,3,:,:) = (etab/etaf)*KX.*KY;
M(4,4,:,:) = -kappaPhibar + (etab/etaf)*(KX2+2*KY2); M(4,5,:,:) = -phibar*KY;
M(5,1,:,:) = (1-phibar)*KX;  M(5,2,:,:) = (1-phibar)*KY;
M(5,3,:,:) = phibar*KX;      M(5,4,:,:) = phibar*KY;

% Decoupling (1): exact DC-mode velocity gauge pin.
M(1,:,1,1) = 0; M(1,1,1,1) = 1;
M(2,:,1,1) = 0; M(2,2,1,1) = 1;

% --- Option A (currently INACTIVE, commented out): decouple pressure at
% EXACTLY the 4 corner modes where kx,ky in {0,Nyquist} simultaneously --
% row 5's entries are all proportional to KX or KY, so it's genuinely,
% fully degenerate ONLY where both vanish at once; everywhere else at
% least one is nonzero, so pressure only partially decouples there -- but
% see the header comment above: at non-corner modes with exactly one axis
% at Nyquist, that "partial decoupling" leaves a near-singular block
% (cond(M) up to ~1.5e6), which caused real GMRES stagnation/blow-up at
% larger theta amplitude. Kept only for regression-testing that failure.
%
% corners = [1,1; 1,Nx/2+1; Ny/2+1,1; Ny/2+1,Nx/2+1];
% for k = 1:size(corners,1)
%     iy = corners(k,1); ix = corners(k,2);
%     M(5,:,iy,ix) = 0; M(:,5,iy,ix) = 0; M(5,5,iy,ix) = 1;
% end

% --- Option B (currently ACTIVE): decouple pressure at every fine mode
% with NO coarse-grid pressure counterpart -- an always-3-wide gap band
% per axis (fine indices Nf/2, Nf/2+1, Nf/2+2, since Nf-Nc=2 exactly
% here): the middle index is the fine grid's own Nyquist mode (no coarse
% counterpart at all); the two flanking indices are where the coarse
% grid's single, ambiguous Nyquist bin would split across on lift/
% restrict. All three are decoupled here -- a minor extra approximation
% for the 2 flanking bins specifically (their content is not actually
% zero in the true system, verified separately), acceptable since GMRES's
% own iterations correct for any remaining preconditioner approximation --
% and, unlike Option A's partial decoupling, keeps cond(M) bounded (~1e5)
% across the whole Nyquist-adjacent band instead of leaving a near-
% singular sliver there.
%
% Do NOT narrow this to decouple only the middle (true-Nyquist) index and
% leave the 2 flanking bins matched -- tried and verified worse, for a
% DIFFERENT reason than Option A's near-singularity (M stays full rank and
% well-conditioned, cond(M)~1e5, even then). The 2 flanking bins are the
% fine grid's two separate halves of the coarse grid's ONE real Nyquist
% pressure dof (liftToFourierImpl SPLITS it across them); the true system
% genuinely couples those two fine modes together through that one shared
% pressure dof. This per-mode-block-diagonal preconditioner structurally
% cannot represent a coupling BETWEEN two different (kx,ky) modes, no
% matter how well-conditioned each mode's own 5x5 block is individually --
% so leaving them "matched" doesn't add a small error, it leaves a
% coupling the preconditioner can never model. Invisible when phi is
% near-uniform (preconditioner is exact there by construction) and forcing
% is weak; verified to reintroduce Option-A-scale error (10-40x) at larger
% theta amplitude once phi/theta gives it enough to actually miss.
%
% Decoupling (2a): pressure's own DC-mode gauge freedom. DC is "matched"
% (a genuine coarse mode, so the gap-band loop below never touches it) but
% KX=KY=0 there too, for a different reason (continuity has no P
% self-term, and every P-coupling entry is proportional to KX or KY) --
% needs its own explicit override, not covered by decoupling (2b).
M(5,:,1,1) = 0; M(:,5,1,1) = 0; M(5,5,1,1) = 1;

% Decoupling (2b): every fine mode with no coarse-grid pressure counterpart.
matchedY = true(Ny,1); matchedY(Ny/2:Ny/2+2) = false;
matchedX = true(Nx,1); matchedX(Nx/2:Nx/2+2) = false;
pressureMatched = matchedY & matchedX';   % Ny x Nx logical

[iyUnmatched, ixUnmatched] = find(~pressureMatched);
for k = 1:numel(iyUnmatched)
    iy = iyUnmatched(k); ix = ixUnmatched(k);
    M(5,:,iy,ix) = 0; M(:,5,iy,ix) = 0; M(5,5,iy,ix) = 1;
end

% M is now full rank at every mode (verified: rank 5/5 everywhere,
% including DC) -- plain inv is safe and cheaper than pinv's SVD.
% Minv = zeros(5,5,Ny,Nx);
% for ix = 1:Nx
%     for iy = 1:Ny
%         Minv(:,:,iy,ix) = inv(M(:,:,iy,ix)); %#ok<MINV>
%     end
% end
Minv= pageinv(M);
end
