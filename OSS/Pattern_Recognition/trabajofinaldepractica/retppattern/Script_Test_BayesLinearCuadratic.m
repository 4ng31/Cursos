%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% 
%    MASTER OSS – UTT (Université de Technologie de Troyes)
%
%    UTN-Facultad Regional Buenos Aires
%
%    Master en Ciencias, Tecnología y Salud (Master STS)
%
%    Especialidad: Optimización y Seguridad de Sistemas (OSS)
%
%                     Año 2015
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
% Fecha: 26/05/2015
%--------------------------------------------------------------------------
%
%--------------------------------------------------------------------------

		
		

clc       ;
clear     ;
close all ; 

NumFeatures = 2 ;

N_C1 = 100 ;
Xtrain_C1 =  5 + sqrt(20)*randn(NumFeatures,N_C1);
Ytrain_C1 =   1 *ones(1,size(Xtrain_C1,2));

N_C2 = 100 ;
Xtrain_C2 = -5 + sqrt(10)* randn(NumFeatures,N_C2);
Ytrain_C2 =  -1 *ones(1,size(Xtrain_C2,2));

Ntrain =  N_C1 + N_C2;


X_train = [Xtrain_C1 Xtrain_C2];
Y_train = [Ytrain_C1 Ytrain_C2];


%figure;
%scatter3(X_train(1,1:end/2),X_train(2,1:end/2),Y_train(1:end/2))
%hold on;
%scatter3(X_train(1,end/2 +1 :end),X_train(2,end/2 +1 :end),Y_train(end/2 +1 :end))
%hold off;


figure;
scatter(X_train(1,1:end/2),X_train(2,1:end/2))
hold on;
scatter(X_train(1,end/2 +1 :end),X_train(2,end/2 +1 :end))
hold off;


X_test = X_train(:,50:150);
Y_test = Y_train(  50:150);


label_C1 = 1 ;
label_C2 = -1 ;

[Labels_L] = func_MLR_linear(X_train, Y_train , X_test, label_C1,label_C2);
error=100 *  sum(Y_test ~=Labels_L)/length(Labels_L)



[Labels_C] = func_MLR_Cuadratic(X_train, Y_train , X_test, label_C1,label_C2);
error=100 *  sum(Y_test ~=Labels_C)/length(Labels_C)




Max_x = max(X_train,[],2) ;
Min_x = min(X_train,[],2) ;

NumPoints_eje_x_1 = 50  ;
NumPoints_eje_x_2 = 50  ;
eje_x_a_1 = linspace(Min_x(1),Max_x(1),NumPoints_eje_x_1);
eje_x_a_2 = linspace(Min_x(2),Max_x(2),NumPoints_eje_x_2);


[grilla_X,grilla_Y] = meshgrid(eje_x_a_1,eje_x_a_2);

grilla_Z_L = zeros(size(grilla_X));

for ix_1 = 1 : NumPoints_eje_x_1
    for ix_2 = 1 : NumPoints_eje_x_2
                         
        x_1_act = grilla_X(ix_1,ix_2);
        x_2_act = grilla_Y(ix_1,ix_2);
        X_test =  [x_1_act ; x_2_act];
        
                
        [Labels_L] = func_MLR_linear(X_train, Y_train , X_test, label_C1,label_C2);
               
        grilla_Z_L(ix_1,ix_2) = Labels_L ;
        

        
    end
end


figure;
pcolor(grilla_X,grilla_Y,grilla_Z_L)
hold on;
scatter(X_train(1,1:end/2),X_train(2,1:end/2),'+G')
scatter(X_train(1,end/2 +1 :end),X_train(2,end/2 +1 :end),'*Y')
hold off;


grilla_Z_C = zeros(size(grilla_X));

for ix_1 = 1 : NumPoints_eje_x_1
    for ix_2 = 1 : NumPoints_eje_x_2
                   
        x_1_act = grilla_X(ix_1,ix_2);
        x_2_act = grilla_Y(ix_1,ix_2);
        X_test =  [x_1_act ; x_2_act];
        
        [Labels_C] = func_MLR_Cuadratic(X_train, Y_train , X_test, label_C1,label_C2);
        
        grilla_Z_C(ix_1,ix_2) = Labels_C ;
        
        
    end
end


figure;
pcolor(grilla_X,grilla_Y,grilla_Z_C)
hold on;
scatter(X_train(1,1:end/2),X_train(2,1:end/2),'+G')
scatter(X_train(1,end/2 +1 :end),X_train(2,end/2 +1 :end),'*Y')
hold off;
