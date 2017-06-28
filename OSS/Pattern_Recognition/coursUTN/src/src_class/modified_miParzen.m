clear
clc
close all
%% <<<<< OS14 - Nicol�s Marx - Parzen >>>>>>>

N=500;
caso=1;
Sig=0.3;
[Y,X]=generar_datos(N,caso);
% Y label 1 -1
% X Coord pts. (x1,x2)
ngrid=50;
%ntrain=length(Y);
[XX,YY] = meshgrid(linspace(-10,10,ngrid),linspace(-10,10,ngrid));

Xtest=XX(:);
Ytest=YY(:);

Xts=[Xtest Ytest];
% Xts coord (Xtest, Ytest)

labels=zeros(length(Xts),1);
lx=length(X);
lxt=length(Xtest);

for i=1:lxt;

    aux=0;
    for j=1:lx;

       aux=aux+Y(j)*(1/(2*pi()*Sig^2)^(0.5))*exp(-(norm(X(j,:)-Xts(i,:)))/(2*Sig^2));

    end
    
    labels(i)=sign(aux);
    %i
    
end
hold on;
gscatter(Xtest,Ytest,labels)


lab=reshape(labels,ngrid,ngrid);
hold on;
contour(XX,YY,lab,[0 0],'LineWidth',2);