% Generar datos de  muestra
[Y,X] = generar_datos(100,3);
% X son las coordenados (x,y)
% Y son los labels del punto (1,-1)

% Crear una grilla de puntos para graficar
ngrid = 100;
[XX,YY] = meshgrid(linspace(-30,30,ngrid),linspace(-30,30,ngrid));


z1=XX(:); z2=YY(:);
Xts=[z1 z2];
[D,I]=pdist2(X,Xts,'euclidean','smallest',5);
labels=zeros(length(Xts),1);
for n=1:length(labels),
    labels(n)=sign(sum(Y(I(:,n))));
end
lab=reshape(labels,ngrid,ngrid);
hold on;contour(XX,YY,lab,[0 0],'LineWidth',2)