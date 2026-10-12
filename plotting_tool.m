function plotting_tool(CoarseMesh,k,parameter,ND) %#ok<INUSL> -- CoarseMesh kept for call-site compatibility, no longer used
Nx=parameter.Nx;
Ny=parameter.Ny;
x=parameter.x;
y=parameter.y;

Ninterx=200;
Nintery=200;
[xinter,~]=fourierGrid(1,Ninterx);
[yinter,~]=fourierGrid(1,Nintery);
[XX,YY]=meshgrid(xinter,yinter);
[X,Y]=meshgrid(x,y);

%% all interpolation below is FFT-based (D.liftP / fourierUpsample2D,
%% O(N log N)) instead of the dense O(N^2) getInterpolationMatrix multiply --
%% exactly the same trigonometric interpolation kernel, verified to ~1e-15.
%% D.liftP handles this codebase's usual coarse=fine-2 pairing (reused
%% directly, not reimplemented); fourierUpsample2D handles upsampling onto
%% the FIXED 200x200 display grid, an arbitrary size ratio D.liftP can't
%% do (it's hardcoded to the +2 pairing).
D=fourierDiffOps(Nx,Ny);
psi=k(1:(Nx-2)*(Ny-2));
phi=D.liftP(psi);                                                     % coarse -> original fine grid (for scatter3 markers)
phi_hires=reshape(fourierUpsample2D(psi, Ny-2, Nx-2, Nintery, Ninterx), Nintery, Ninterx);  % coarse -> display grid, directly
Uf1=k((Nx-2)*(Ny-2)+1:2*Nx*Ny-2*Nx-2*Ny+4);
Uf2=k(2*Nx*Ny-2*Nx-2*Ny+5:3*Nx*Ny-2*Nx-2*Ny+4);
P=k(3*Nx*Ny-2*Nx-2*Ny+5:4*Nx*Ny-4*Nx-4*Ny+7);
Up1=k(4*Nx*Ny-4*Nx-4*Ny+8:5*Nx*Ny-4*Nx-4*Ny+7);
Up2=k(5*Nx*Ny-4*Nx-4*Ny+8:6*Nx*Ny-4*Nx-4*Ny+7);
theta=k(6*Nx*Ny-4*Nx-4*Ny+8:7*Nx*Ny-4*Nx-4*Ny+7);

% P (length Np=NP-1) is the zero-mean pressure gauge solveVelocity[FFT].m
% actually solves for -- mean(Pfull_coarse)=PrefPt exactly, NOT a
% point-pin at some reference index. Same embed as velocityPinv.m's
% dPfull=[dP;-sum(dP)] and test_velocity_fft_accuracy.m's own
% reconstruction: [P;-sum(P)] spans the same zero-mean subspace, then the
% PrefPt offset is added once here.
Pfull_coarse = [P; -sum(P)] + parameter.PrefPt;
Pext=D.liftP(Pfull_coarse);                                           % coarse -> original fine grid (for scatter3 markers)
Pext_hires=reshape(fourierUpsample2D(Pfull_coarse, Ny-2, Nx-2, Nintery, Ninterx), Nintery, Ninterx);  % coarse -> display grid, directly

Uf1_hires=reshape(fourierUpsample2D(Uf1, Ny, Nx, Nintery, Ninterx), Nintery, Ninterx);
Uf2_hires=reshape(fourierUpsample2D(Uf2, Ny, Nx, Nintery, Ninterx), Nintery, Ninterx);
Up1_hires=reshape(fourierUpsample2D(Up1, Ny, Nx, Nintery, Ninterx), Nintery, Ninterx);
Up2_hires=reshape(fourierUpsample2D(Up2, Ny, Nx, Nintery, Ninterx), Nintery, Ninterx);
theta_hires=reshape(fourierUpsample2D(theta, Ny, Nx, Nintery, Ninterx), Nintery, Ninterx);
if ND==3
    figure  
    subplot(4,2,1)
    scatter3(X(:),Y(:),Uf1,20,Uf1,'filled','k')
    hold on
    mesh(XX,YY,Uf1_hires)
    colormap jet
    colorbar
    title("Uf1")

    subplot(4,2,2)
    scatter3(X(:),Y(:),Uf2,20,Uf2,'filled','k')
    hold on
    mesh(XX,YY,Uf2_hires)
    colorbar
    title("Uf2")


    subplot(4,2,3)
    scatter3(X(:),Y(:),Up1,20,Up1,'filled','k')
    hold on
    mesh(XX,YY,Up1_hires)
    colormap jet
    colorbar
    title("Up1")


    subplot(4,2,4)
    scatter3(X(:),Y(:),Up2,20,Up2,'filled','k')
    hold on
    mesh(XX,YY,Up2_hires)
    colormap jet
    colorbar
    title("Up2")
    
    
    subplot(4,2,5)
    scatter3(X(:),Y(:),phi,20,phi,'filled','k')
    hold on
    mesh(XX,YY,phi_hires)    
    colormap jet  
    colorbar
    title("phi")
    
    % 
    subplot(4,2,6)
    scatter3(X(:),Y(:),theta,20,theta,'filled','k')
    hold on
    mesh(XX,YY,theta_hires)
    colormap jet
    colorbar
    title("theta")


    subplot(4,2,7)
    scatter3(X(:),Y(:),Pext,20,Pext,'filled','k')
    hold on
    mesh(XX,YY,Pext_hires)
    colormap jet
    colorbar
    title("P")
elseif ND==2  
    figure('units','normalized','outerposition',[0 0 1 1]);
    set(gcf, 'Position', [0 0 1 1]); % both even
    axis off; % optional (removes ticks, labels)
    tiledlayout(2,3,'TileSpacing','compact','Padding','compact')
    
    nexttile
    U=Uf1_hires;
    V=Uf2_hires;
    step = 6;
    quiver(XX(1:step:end,1:step:end), ...
           YY(1:step:end,1:step:end), ...
           U(1:step:end,1:step:end), ...
           V(1:step:end,1:step:end),'k')
    
    axis image
    xlim([-1 1])
    ylim([-1 1])
    title("U_{fluid}")
    
    nexttile
    U=Up1_hires;
    V=Up2_hires;
    step = 6;
    quiver(XX(1:step:end,1:step:end), ...
           YY(1:step:end,1:step:end), ...
           U(1:step:end,1:step:end), ...
           V(1:step:end,1:step:end),'k')
    
    axis image
    xlim([-1 1])
    ylim([-1 1])
    title("U_{particle}")
    
    nexttile
    contourf(XX,YY,Pext_hires,50,'LineColor','none')
    colormap jet
    colorbar
    title("P")
    axis image

    nexttile
    contourf(XX,YY,phi_hires,50,'LineColor','none')
    colormap jet
    colorbar 
    axis image
    xlim([-1 1])
    ylim([-1 1])
    title("\phi")
    
    nexttile
    U = cos(theta_hires);
    V = sin(theta_hires);
    step = 6;   
    quiver(XX(1:step:end,1:step:end), ...
           YY(1:step:end,1:step:end), ...
           U(1:step:end,1:step:end), ...
           V(1:step:end,1:step:end),'k')
    
    axis image
    xlim([-1 1])
    ylim([-1 1])
    title("\theta")
    

    nexttile
    U = Uf1_hires;
    V = Uf2_hires;
    contourf(XX,YY,sqrt(U.^2+V.^2),50,'LineColor','none')
    colormap jet
    colorbar 
    hold on 
    step = 6;
    
    quiver(XX(1:step:end,1:step:end), ...
           YY(1:step:end,1:step:end), ...
           U(1:step:end,1:step:end), ...
           V(1:step:end,1:step:end),'k')
    
    axis image
    xlim([-1 1])
    ylim([-1 1])
    title("||U_{fluid}||")
end
% %1d plot
% [X,Y]=meshgrid(x,y);
% X=X(:);Y=Y(:);
% IDXx0=find(X==0);
% IDYy0=find(Y==0);
% 
% XX=XX(:);YY=YY(:);
% IDXXx0=find(XX==0);
% IDYYy0=find(YY==0);
% 
% Uf1Uf1=interMat*Uf1;
% Uf2Uf2=interMat*Uf2;
% phiphi=interMat*phi;
% 
% figure 
% subplot(3,2,1)
% plot(Y(IDXx0),Uf1(IDXx0),'k*')
% xlabel('y')
% ylabel('Uf1')
% hold on
% plot(YY(IDXXx0),Uf1Uf1(IDXXx0))
% subplot(3,2,2)
% plot(X(IDYy0),Uf1(IDYy0),'k*')
% hold on
% plot(XX(IDYYy0),Uf1Uf1(IDYYy0))
% xlabel('x')
% ylabel('Uf1')
% 
% subplot(3,2,3)
% plot(Y(IDXx0),Uf2(IDXx0),'k*')
% hold on
% plot(YY(IDXXx0),Uf2Uf2(IDXXx0))
% xlabel('y')
% ylabel('Uf2')
% 
% subplot(3,2,4)
% plot(X(IDYy0),Uf2(IDYy0),'k*')
% hold on
% plot(XX(IDYYy0),Uf2Uf2(IDYYy0))
% xlabel('x')
% ylabel('Uf2')
% 
% subplot(3,2,5)
% plot(Y(IDXx0),phi(IDXx0),'k*')
% hold on
% plot(YY(IDXXx0),phiphi(IDXXx0))
% xlabel('y')
% ylabel('phi')
% 
% subplot(3,2,6)
% plot(X(IDYy0),phi(IDYy0),'k*')
% hold on
% plot(XX(IDYYy0),phiphi(IDYYy0))
% xlabel('x')
% ylabel('phi')
end
