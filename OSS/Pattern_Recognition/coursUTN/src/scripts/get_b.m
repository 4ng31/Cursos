function [b_opt] = get_b(NET,alpha,X,Y,tol)
if nargin < 5,
    tol = .001;
end
C = NET.C;
kernel_type = NET.ker;
kernel_param = NET.kerpar;

%Los vectores soportes del lado positivo
idx_sp_plus   = find( (alpha < (C-tol)) & (alpha > tol) & (Y ==+1));
%Los vectores sopoertes del lado negativo
idx_sp_minus  = find( (alpha < (C-tol)) & (alpha > tol) & (Y == -1));
% Todos los vectores sopoertes
idx_sp        = find( (alpha > tol) );    
n            = length(idx_sp);

% Calculo del valor de b para el margen positivo
n_plus       = length(idx_sp_plus);
b_plus       = zeros(n_plus,1);
for m=1:n_plus,
   index_m = idx_sp_plus(m);  
   Sumatoria = 0;
   for i=1:n,
     index_i   = idx_sp(i);  
     Sumatoria = Sumatoria + alpha(index_i)*Y(index_i)*kernelpt(kernel_type, kernel_param, X(index_i,:), X(index_m,:));
   end
   b_plus(m) = 1 - Sumatoria;
end
b_opt_plus      = mean(b_plus);

% Calculo del valor de b para el margen negativo
n_minus         = length(idx_sp_minus);
b_minus         = zeros(n_minus,1);
for m=1:n_minus,
   index_m = idx_sp_minus(m);  
   Sumatoria = 0;
   for i=1:n,
     index_i   = idx_sp(i);  
     Sumatoria = Sumatoria + alpha(index_i)*Y(index_i)*kernel(kernel_type, kernel_param, X(index_i,:), X(index_m,:));
   end
   b_minus(m) = -1 - Sumatoria;
end
b_opt_minus      = mean(b_minus);


b_opt            = ( ( n_plus * b_opt_plus ) +...
                   ( n_minus * b_opt_minus ) ) / ( n_plus + n_minus );
