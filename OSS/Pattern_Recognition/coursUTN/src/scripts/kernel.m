function [K]=kernel(ker, param, X1, X2)
% funcion optimizada para calculos matriciales.
% Devuelve la matrz K completa.
[n1,p]=size(X1);
[n2,p]=size(X2);
switch lower(ker)
    case 'poly',    K = (X1*X2' + 1).^param;
    case 'rbf',
                    if p==1;
                        MX1 = (X1'.*X1')';
                        MX2 = (X2'.*X2')';
                        K = MX1*ones(1,n2) + ones(n1,1)*MX2' - 2*X1*X2';
                        K = exp((-K)/(2*param*param)); 
                    else            
                        MX1 = sum(X1'.*X1')';
                        MX2 = sum(X2'.*X2')';
                        K = MX1*ones(1,n2) + ones(n1,1)*MX2' - 2*X1*X2';
                        K = exp((-K)/(2*param*param)); 
                    end;    
    case 'idem',
		    K = X1*X2';
end