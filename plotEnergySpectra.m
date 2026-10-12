function plotEnergySpectra(kk, tt, Nx, Ny, CoarseMesh, parameter) %#ok<INUSL> -- CoarseMesh kept for call-site compatibility, no longer used
% Plots the fluid, particle, elastic, mixture, and active-stress-divergence
% energy spectra, averaged over time (all snapshots in kk / tt), in one figure.
% kk:         state matrix (columns = snapshots)
% tt:         time vector
% Nx, Ny:     grid sizes
% parameter:  struct from WolgemuthCase.m (etaf, R, A, b, boolpusher)
%
% phi's coarse->fine lift and the active-stress term's derivatives are
% FFT-based (D.liftP / D.DXf/D.DYf, O(N log N)) rather than the dense
% CoarseMesh.interMatP multiply / parameter.DX,DY matrices -- same
% trigonometric interpolation/differentiation kernel, see plotting_tool.m.

etaf=parameter.etaf;
R=parameter.R;
A=parameter.A;
Ax=A;
Ay=A;
b=parameter.b;
alphaSign=-2*parameter.boolpusher+1;

D = fourierDiffOps(Nx, Ny);

idx_psi   = 1 : (Nx-2)*(Ny-2);
idx_Uf1   = (Nx-2)*(Ny-2)+1     : 2*Nx*Ny-2*Nx-2*Ny+4;
idx_Uf2   = 2*Nx*Ny-2*Nx-2*Ny+5 : 3*Nx*Ny-2*Nx-2*Ny+4;
idx_Up1   = 4*Nx*Ny-4*Nx-4*Ny+8 : 5*Nx*Ny-4*Nx-4*Ny+7;
idx_Up2   = 5*Nx*Ny-4*Nx-4*Ny+8 : 6*Nx*Ny-4*Nx-4*Ny+7;
idx_theta = 6*Nx*Ny-4*Nx-4*Ny+8 : 7*Nx*Ny-4*Nx-4*Ny+7;

k_max = floor(min(Nx,Ny)/2);

% integer mode indices, shared by all four spectra (all live on the Nx x Ny grid)
kx_idx = [0:Nx/2, -Nx/2+1:-1];
ky_idx = [0:Ny/2, -Ny/2+1:-1];
[KXi, KYi] = meshgrid(kx_idx, ky_idx);
Kmag_idx = round(sqrt(KXi.^2 + KYi.^2));
KX = (pi/Ax) * KXi;
KY = (pi/Ay) * KYi;

nSnap = size(kk, 2);
E2D_fluid     = zeros(Ny, Nx);
E2D_particle  = zeros(Ny, Nx);
E2D_elastic   = zeros(Ny, Nx);
E2D_mixture   = zeros(Ny, Nx);
E2D_activeDiv = zeros(Ny, Nx);

for s = 1:nSnap
    psi   = kk(idx_psi, s);
    Uf1   = kk(idx_Uf1, s);
    Uf2   = kk(idx_Uf2, s);
    Up1   = kk(idx_Up1, s);
    Up2   = kk(idx_Up2, s);
    theta = kk(idx_theta, s);
    phi   = D.liftP(psi);
    d1    = cos(theta);
    d2    = sin(theta);

    % fluid kinetic energy spectrum: E(k) = 0.5*(|FFT(Uf1)|^2+|FFT(Uf2)|^2)
    Uf1hat = fft2(reshape(Uf1, Ny, Nx));
    Uf2hat = fft2(reshape(Uf2, Ny, Nx));
    E2D_fluid = E2D_fluid + 0.5*(abs(Uf1hat).^2 + abs(Uf2hat).^2) / (Nx*Ny)^2;

    % particle kinetic energy spectrum: E(k) = 0.5*(|FFT(Up1)|^2+|FFT(Up2)|^2)
    Up1hat = fft2(reshape(Up1, Ny, Nx));
    Up2hat = fft2(reshape(Up2, Ny, Nx));
    E2D_particle = E2D_particle + 0.5*(abs(Up1hat).^2 + abs(Up2hat).^2) / (Nx*Ny)^2;

    % elastic energy spectrum: E(k) = 0.5*|k|^2*|FFT(theta)|^2
    Thetahat = fft2(reshape(theta, Ny, Nx));
    E2D_elastic = E2D_elastic + 0.5 * (KX.^2 + KY.^2) .* abs(Thetahat).^2 / (Nx*Ny)^2;

    % mixture kinetic energy spectrum: spectrum of phi*Up^2 + (1-phi)*Uf^2
    mixKE = phi.*(Up1.^2 + Up2.^2) + (1-phi).*(Uf1.^2 + Uf2.^2);
    MixKEhat = fft2(reshape(mixKE, Ny, Nx));
    E2D_mixture = E2D_mixture + abs(MixKEhat).^2 / (Nx*Ny)^2;

    % active stress divergence term in the fluid momentum eq (getEqSys.m):
    % nabla.(C*d(x)d), C = alpha*ksi*(A*b/etaf), d = (cos(theta),sin(theta))
    ksi = 2*etaf/R^2 .* phi;
    C = alphaSign .* ksi .* (A*b/etaf);
    termX = D.DXf(C.*d1.*d1) + D.DYf(C.*d1.*d2);
    termY = D.DXf(C.*d2.*d1) + D.DYf(C.*d2.*d2);
    TermXhat = fft2(reshape(termX, Ny, Nx));
    TermYhat = fft2(reshape(termY, Ny, Nx));
    E2D_activeDiv = E2D_activeDiv + 0.5*(abs(TermXhat).^2 + abs(TermYhat).^2) / (Nx*Ny)^2;
end

% time-average over all snapshots
E2D_fluid     = E2D_fluid     / nSnap;
E2D_particle  = E2D_particle  / nSnap;
E2D_elastic   = E2D_elastic   / nSnap;
E2D_mixture   = E2D_mixture   / nSnap;
E2D_activeDiv = E2D_activeDiv / nSnap;

E_fluid     = radialBin(E2D_fluid,     Kmag_idx, k_max);
E_particle  = radialBin(E2D_particle,  Kmag_idx, k_max);
E_elastic   = radialBin(E2D_elastic,   Kmag_idx, k_max);
E_mixture   = radialBin(E2D_mixture,   Kmag_idx, k_max);
E_activeDiv = radialBin(E2D_activeDiv, Kmag_idx, k_max);

k_plot = (1:k_max)'; % drop kr=0 (log(0) undefined)

figure('units','normalized','outerposition',[0 0 1 1]);
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

nexttile;
plotOnePanel(k_plot, E_fluid(2:end), tt(1), tt(end), true);
ylabel('E(k) = 0.5 \Sigma (|FFT(u)|^2+|FFT(v)|^2)');
title('Fluid kinetic energy spectrum');

nexttile;
plotOnePanel(k_plot, E_particle(2:end), tt(1), tt(end), true);
ylabel('E(k) = 0.5 \Sigma (|FFT(u_p)|^2+|FFT(v_p)|^2)');
title('Particle kinetic energy spectrum');

nexttile;
plotOnePanel(k_plot, E_elastic(2:end), tt(1), tt(end), true);
ylabel('E(k) = 0.5 \Sigma |k|^2 |FFT(\theta)|^2');
title('Elastic energy spectrum');

nexttile;
plotOnePanel(k_plot, E_mixture(2:end), tt(1), tt(end), true);
ylabel('E(k) = \Sigma |FFT(\phi U_p^2+(1-\phi)U_f^2)|^2');
title('Mixture kinetic energy spectrum');

nexttile;
plotOnePanel(k_plot, E_activeDiv(2:end), tt(1), tt(end), true);
ylabel('E(k) = 0.5 \Sigma |FFT(active stress divergence)|^2');
title('Active stress divergence spectrum (fluid momentum eq.)');

sgtitle(sprintf('Time-averaged energy spectra (t \\in [%.4f, %.4f], %d snapshots)', tt(1), tt(end), numel(tt)));
end

function E_radial = radialBin(E2D, Kmag_idx, k_max)
E_radial = zeros(k_max+1, 1);
for kr = 0:k_max
    E_radial(kr+1) = sum(E2D(Kmag_idx == kr));
end
end

function plotOnePanel(k_plot, E_plot, t_start, t_end, fitSlope)
loglog(k_plot, E_plot, 'o-', 'MarkerSize', 4, 'DisplayName', sprintf('t\\in[%.4f,%.4f]', t_start, t_end));
hold on;
if fitSlope
    [pBest, R2Best, cBest, kLo, kHi] = fitStraightestSlope(k_plot, E_plot);
    if isfinite(pBest)
        % Drawn a bit past the window it was actually fit over (not the
        % whole panel -- that overshot badly) and pushed outward from the
        % data by offsetFactor (same slope, shifted intercept): a fit line
        % sitting exactly on top of the curve it was fit to is nearly
        % indistinguishable from the curve itself, so it's extended by a
        % margin proportional to the fitted window's own (log-)width and
        % offset vertically, reading as its own short reference line
        % alongside that stretch instead of tracing over it.
        offsetFactor = 4;    % exterior displacement, in linear E units -- purely
                              % visual separation, doesn't change the reported slope
        marginFrac = 0.4;    % extend by 40% of the fit window's own log-width on each side
        span = log(kHi) - log(kLo);
        kLine = exp([log(kLo) - marginFrac*span; log(kHi) + marginFrac*span]);
        ELine = offsetFactor * exp(polyval(cBest, log(kLine)));
        loglog(kLine, ELine, 'r-', 'LineWidth', 2.5, ...
            'DisplayName', sprintf('fit: k^{%.2f} (R^2=%.3f)', pBest, R2Best));
    end
end
xlabel('|k|');
legend show;
grid on;
end

function [pBest, R2Best, cBest, kLo, kHi] = fitStraightestSlope(k, E)
% Finds the contiguous sub-range of (k,E) whose log-log representation is
% closest to a straight line (highest R^2 of a linear least-squares fit
% among windows spanning at least minPoints samples), and returns that
% fit's [slope, intercept] (cBest) plus the window's own k-bounds (kLo,
% kHi) so the caller can draw the line a controlled amount past the
% window itself, rather than being stuck with either the bare fitted
% stretch or the whole panel. This is the power-law region an
% energy-spectrum slope is normally quoted over, found automatically
% instead of eyeballing where the loglog curve looks straight or
% overlaying fixed reference slopes (k^-1/-2/-3, the previous approach
% here) for visual comparison only.
k = k(:); E = E(:);
valid = isfinite(k) & isfinite(E) & (k > 0) & (E > 0);
k = k(valid); E = E(valid);
logk = log(k); logE = log(E);
N = numel(logk);

pBest = NaN; R2Best = -Inf; cBest = [NaN NaN]; kLo = NaN; kHi = NaN;
if N < 5
    return   % not enough valid (finite, positive) points to fit anything meaningful
end

% Minimum window width: at least a quarter of the available range, so a
% trivial 2-3 point segment can't "win" just for being perfectly (and
% meaninglessly) collinear.
minPoints = max(5, round(0.25*N));
for i = 1:(N-minPoints+1)
    for j = (i+minPoints-1):N
        x = logk(i:j); y = logE(i:j);
        if var(x) == 0 || var(y) == 0
            continue   % degenerate window (e.g. sitting on a flat noise
                       % floor) -- not a meaningful power-law slope, skip
                       % rather than let a trivial "perfect" flat fit win
        end
        c = polyfit(x, y, 1);
        ssRes = sum((y - polyval(c, x)).^2);
        ssTot = sum((y - mean(y)).^2);
        R2 = 1 - ssRes/ssTot;
        if R2 > R2Best
            R2Best = R2; cBest = c; kLo = k(i); kHi = k(j);
        end
    end
end

pBest = cBest(1);
end
