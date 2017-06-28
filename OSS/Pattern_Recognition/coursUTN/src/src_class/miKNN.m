clear
clc
close all
%% <<<<< OS14 - Nicolás Marx - KNN  >>>>>>>

N=100;
caso=2;
K=3;
[Y,X]=generar_datos(N,caso);
ngrid=1000;
ntrain=length(Y);
[XX,YY] = meshgrid(linspace(-30,30,ngrid),linspace(-30,30,ngrid));

Xtest=XX(:);
Ytest=YY(:);

Xts=[Xtest Ytest];
labels=zeros(length(Xts),1);

% Sin el PDIST2
Knn=Inf(length(Xtest),K,2);
for idx=1:length(Xtest);
    
    for j=1:ntrain;
        aux=norm(X(j,:)-Xts(idx,:));
        [maxd,posmax]=max(Knn(idx,:,1));
        if aux<maxd;
            Knn(idx,posmax,1)=aux;
            Knn(idx,posmax,2)=j;
        end;
    end;
     
    labels(idx)=sign(sum(Y(Knn(idx,:,2))));
        
end;



%---------------
%Acá termina sin el PDIST2
%----------------





%{
Con el PDIST2
[D,I]=pdist2(X,Xts,'euclidean','smallest',K);



for n=1:length(labels);
    labels(n)=sign(sum(Y(I(:,n))));
end

%---------------
%Acá termina conn el PDIST2
%----------------
%}


lab=reshape(labels,ngrid,ngrid);
hold on;
contour(XX,YY,lab,[0 0],'LineWidth',2);