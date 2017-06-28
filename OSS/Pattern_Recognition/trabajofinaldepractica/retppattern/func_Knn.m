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
% Fecha: 20/05/2015
%--------------------------------------------------------------------------
%
%--------------------------------------------------------------------------

function [ Y_test ] = func_Knn(X_train,Y_train,k,X_test)
    

NumExamples = size(X_train,2);
NumPatternToTEst = size(X_test,2);

Y_test = NaN * zeros(1,NumPatternToTEst);


  for ix = 1 : NumPatternToTEst
    

        Val_min = inf*ones(1,k) ;
        Ind_min = zeros(1,k) ;
               
        for i_test = 1 : NumExamples
                                   
            dist = norm( X_test(:,ix) - X_train(:,i_test) );
            
            
            [val ,  ind] = max(Val_min,[],2) ;
            
            if dist < val 
                
                Val_min(ind) = dist  ;
                Ind_min(ind) = i_test;                
                
            end            
        end
        
        % esto funciona porque las etiquetas de clases son -1 y +1
        result = sign(sum(Y_train(Ind_min))) ;
        
        Y_test(ix) = result ;
        
  end
  

















