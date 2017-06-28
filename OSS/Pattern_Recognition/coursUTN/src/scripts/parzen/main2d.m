Num=10;
[Y,X] = generar_datos(Num,1);
% Crear una grilla de puntos para graficar
ngrid = 1000;
[XX,YY] = meshgrid(linspace(-30,30,ngrid),linspace(-30,30,ngrid));
Xts=;


%Sumar probabilidad
N=size(Xts);
hn=h1/sqrt(N);
[X Xi]=meshgrid(x,xi);
p=sum(f((X-Xi)/hn)/hn)/N;

h2=0.5;
p=parzen2d(Xts,X,h2,);
figure,plot(x,p);
title('parzen h2=0.5');
ylabel('N=%d',Num);
