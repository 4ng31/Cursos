close all; clear;
 
% Generar datos de  muestra
h = [1 0.6 0.15];  % Ancho Ventana
n = 1000; % Numero de Muestras

[Y,X] = generar_datos(1000,1);
% X puntos donde se estimara la densidad usando ventana parzen
% Y son los labels del punto (1,-1)

% Crear una grilla de puntos para graficar
ngrid = 1000;
[XX,YY] = meshgrid(linspace(-30,30,ngrid),linspace(-30,30,ngrid));

for i=1:n
  u=X(:);
  d = length(u);
  y(i) = exp(-(u'*u)/2)/((2*pi)^(d/2));
end

if (p1 >= p2)
  class = 1;
else
  class = 2;
end
