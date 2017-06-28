%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% 
%    MASTER OSS � UTT (Universit� de Technologie de Troyes)
%
%    UTN-Facultad Regional Buenos Aires
%
%    Master en Ciencias, Tecnolog�a y Salud (Master STS)
%
%    Especialidad: Optimizaci�n y Seguridad de Sistemas (OSS)
%
%                     A�o 2015
%
%--------------------------------------------------------------------------
%
%    Pattern Recognition (OS2)
%
%    Profesores:
%                Dr. Beauseroy
%                Dr. Tomassi
%               
%--------------------------------------------------------------------------
% Alumno: Martinez Garbino, Lucio Jose
% Fecha: 20/05/2015
%--------------------------------------------------------------------------
%
%--------------------------------------------------------------------------


clc       ;
clear     ;
fclose all;
close all ; 
close;


rng default;
NumFeatures = 2 ;

N_C1 = 50 ;
Xtrain_C1 =  5 + sqrt(20)*randn(NumFeatures,N_C1);
Ytrain_C1 =   1 *ones(1,size(Xtrain_C1,2));

N_C2 = 50 ;
Xtrain_C2 = -5 + sqrt(10)* randn(NumFeatures,N_C2);
Ytrain_C2 =  -1 *ones(1,size(Xtrain_C2,2));

Ntrain =  N_C1 + N_C2;


X_train = [Xtrain_C1 Xtrain_C2];
Y_train = [Ytrain_C1 Ytrain_C2];



figure;
scatter(X_train(1,1:end/2),X_train(2,1:end/2))
hold on;
scatter(X_train(1,end/2 +1 :end),X_train(2,end/2 +1 :end))
hold off;


Sigma = 3 ;


Max_x = max(X_train,[],2) ;
Min_x = min(X_train,[],2) ;

NumPoints_eje_x_1 = 25  ;
NumPoints_eje_x_2 = 25  ;
eje_x_a_1 = linspace(Min_x(1),Max_x(1),NumPoints_eje_x_1);
eje_x_a_2 = linspace(Min_x(2),Max_x(2),NumPoints_eje_x_2);


[grilla_X,grilla_Y] = meshgrid(eje_x_a_1,eje_x_a_2);

grilla_Z = zeros(numel(eje_x_a_1),numel(eje_x_a_2));


h1=waitbar(0,'Calculando');

NumIteraciones = NumPoints_eje_x_1 * NumPoints_eje_x_2;
CountIter = 0 ;

tic
for ix_1 = 1 : NumPoints_eje_x_1
    for ix_2 = 1 : NumPoints_eje_x_2              
        x_1_act = grilla_X(ix_1,ix_2);
        x_2_act = grilla_Y(ix_1,ix_2);
        X_test =  [x_1_act ; x_2_act];
        for i=1:Ntrain            
            [ Label ] = func_Parzen(X_train,Y_train,Sigma,X_test);            
            grilla_Z(ix_1,ix_2) = Label ;
        end  
       CountIter=CountIter+1;
       waitbar(CountIter/NumIteraciones);
    end
end
toc
close(h1);

figure;
pcolor(grilla_X,grilla_Y,grilla_Z)
hold on;
scatter(X_train(1,1:end/2),X_train(2,1:end/2),'+G')
scatter(X_train(1,end/2 +1 :end),X_train(2,end/2 +1 :end),'*Y')
hold off;
