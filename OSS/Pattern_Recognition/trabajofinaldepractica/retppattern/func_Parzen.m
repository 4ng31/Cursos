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

function [ Y_test ] = func_Parzen(X_train,Y_train,Sigma,X_test)

NumFeatures = size(X_train,1);
NumPatternTrain = size(X_train,2);
NumPatternToTest = size(X_test,2);

size(X_train)
size(X_test)

Y_test = NaN * zeros(1,NumPatternToTest);

Mat_Sigma = Sigma * eye(NumFeatures);
%Det_Sigma = det(Mat_Sigma);
%Inv_Sigma = inv(Mat_Sigma);

  for i_test = 1 : NumPatternToTest
        %Gauss=mvnpdf(X,MU,SIGMA);
        %F = mvnpdf([X1(:) X2(:)],mu,Sigma);
        Gauss=mvnpdf( X_test(:,i_test)' , X_train(:,:)' , Mat_Sigma ) ;
        Acum = Y_train * Gauss ;
        Y_test(i_test) =  sign(Acum);
  end
