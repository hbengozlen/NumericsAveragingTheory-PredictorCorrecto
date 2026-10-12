function [interMatx,interMaty,interMat] = getInterpolationMatrix(xx1,yy1,Nx1,Ny1,xx2,yy2,Nx2,Ny2)


xx1Re=reshape(xx1,Ny1,Nx1);
yy1Re=reshape(yy1,Ny1,Nx1);
x1=xx1Re(1,:);
y1=yy1Re(:,1);
Ax=max(abs(x1));
Ay=max(abs(y1));
xx2Re=reshape(xx2,Ny2,Nx2);
yy2Re=reshape(yy2,Ny2,Nx2);
x2=xx2Re(1,:);
y2=yy2Re(:,1);

interMatx=zeros(Nx2,Nx1);
for i=1:Nx2
    for j=1:Nx1
        if abs(x2(i)-x1(j)) <1e-15
            interMatx(i,j)=1;
        else
            tmp=pi/Ax*(x2(i)-x1(j));
            interMatx(i,j)=sin(Nx1/2*tmp)/Nx1/tan(tmp/2);
        end
    end
end

interMaty=zeros(Ny2,Ny1);
for i=1:Ny2
    for j=1:Ny1
        if abs(y2(i)-y1(j)) <1e-15
            interMaty(i,j)=1;
        else
            tmp=pi/Ay*(y2(i)-y1(j));
            interMaty(i,j)=sin(Ny1/2*tmp)/Ny1/tan(tmp/2);
        end
    end
end

%kronecker product
interMat=kron(interMatx,interMaty);