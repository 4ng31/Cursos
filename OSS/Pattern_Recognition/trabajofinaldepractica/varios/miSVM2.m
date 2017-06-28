function [labels_svm]=miSVM2(Xtrain_res,Ytrain_res,Xval,tipoKernel,paramKernel,C)
options=optimset('Display','off');
clases=unique(Ytrain_res);
if length(clases)~=2,
    error('Este clasificador SVM solo funciona para dos clases');
end

ntrain=length(Ytrain_res(:,1));

Ytrain_res(Ytrain_res==clases(1))=-1;
Ytrain_res(Ytrain_res==clases(2))=1;

ntest=length(Xval(:,1));    

H=NaN(ntrain);
if nargin<6
tipoKernel='gaussiano'; %opciones: 'sin kernel', 'gaussiano', 'polinomico'
end

switch lower(tipoKernel)
    case 'sin kernel'
        for i=1:ntrain
            for j=i:ntrain
                H(i,j)=Xtrain_res(i,:)*Xtrain_res(j,:)';
                H(j,i)=H(i,j);
            end
        end
    case 'gaussiano'
        for i=1:ntrain
            for j=i:ntrain
                dif=norm(Xtrain_res(i,:)-Xtrain_res(j,:));
                H(i,j)=exp(-0.5*(dif*dif')/(paramKernel^2));
                H(j,i)=H(i,j);
            end
        end
    case 'polinomico'
        for i=1:ntrain
            for j=i:ntrain
        H(i,j)=(1+Xtrain_res(i,:)-Xtrain_res(j,:)')^paramKernel;
        H(j,i)=H(i,j);
            end
        end
end    

f=-ones(ntrain,1); A=[]; b=[]; Aeq=Ytrain_res'; beq=0; LB=zeros(ntrain,1); UB=C*ones(ntrain,1); X0=zeros(ntrain ,1);

a=quadprog(H,f,A,b,Aeq,beq,LB,UB,X0,options);

tol=0.001;

indice_vs_pos=find( (a < (C-tol)) & (a > tol) & (Ytrain_res == clases(1))); %Los vectores soportes del lado positivo

indice_vs_neg=find( (a < (C-tol)) & (a > tol) & (Ytrain_res == clases(2))); %Los vectores soportes del lado negativo

indice_vs= find( (a > tol) );  % Todos los vectores soportes  

num_vs = length(indice_vs); %Cantidad de vectores soportes


n_pos = length(indice_vs_pos); %Cantidad de vectores soportes del margen positivo
b_pos = zeros(n_pos,1);

n_neg = length(indice_vs_neg); %Cantidad de vectores soportes del margen positivo
b_neg = zeros(n_neg,1);


switch lower(tipoKernel)
    case 'sin kernel'
       for m=1:n_pos %Recorre todos los VS positivos
           idx_vs_pos = indice_vs_pos(m);  
           Sum = 0;
               for i=1:num_vs, %Recorre todos los VS
                     idx_vs   = indice_vs(i);  
                     Sum = Sum + a(idx_vs)*Ytrain_res(idx_vs)*(Xtrain_res(idx_vs,:)*Xtrain_res(idx_vs_pos,:)');
               end
           b_pos(m) = 1 - Sum;
       end
       
       for m=1:n_neg %Recorre todos los VS negativos
           idx_vs_neg = indice_vs_neg(m);  
           Sum = 0;
               for i=1:num_vs, %Recorre todos los VS
                     idx_vs   = indice_vs(i);  
                     Sum = Sum + a(idx_vs)*Ytrain_res(idx_vs)*(Xtrain_res(idx_vs,:)*Xtrain_res(idx_vs_neg,:)');
               end
         b_neg(m) = -1 - Sum;
       end
       
        
    case 'gaussiano'
       
        for m=1:n_pos %Recorre todos los VS positivos
           idx_vs_pos = indice_vs_pos(m);  
           Sum = 0;
               for i=1:num_vs, %Recorre todos los VS
                     idx_vs   = indice_vs(i);  
                     dif2=norm(Xtrain_res(idx_vs,:)-Xtrain_res(idx_vs_pos,:));
                     Sum = Sum + a(idx_vs)*Ytrain_res(idx_vs)*(exp(-0.5*(dif2*dif2')/(paramKernel^2)));
               end
           b_pos(m) = 1 - Sum;
       end
       
       for m=1:n_neg %Recorre todos los VS negativos
           idx_vs_neg = indice_vs_neg(m);  
           Sum = 0;
               for i=1:num_vs, %Recorre todos los VS
                     idx_vs   = indice_vs(i);  
                     dif2=norm(Xtrain_res(idx_vs,:)-Xtrain_res(idx_vs_neg,:));
                     Sum = Sum + a(idx_vs)*Ytrain_res(idx_vs)*(exp(-0.5*(dif2*dif2')/(paramKernel^2)));
               end
         b_neg(m) = -1 - Sum;
       end
        
    case 'polinomico'
        
       for m=1:n_pos %Recorre todos los VS positivos
           idx_vs_pos = indice_vs_pos(m);  
           Sum = 0;
               for i=1:num_vs, %Recorre todos los VS
                     idx_vs   = indice_vs(i);  
                     Sum = Sum + a(idx_vs)*Ytrain_res(idx_vs)*((1+Xtrain_res(idx_vs,:)-Xtrain_res(idx_vs_pos,:)')^paramKernel);
               end
           b_pos(m) = 1 - Sum;
       end
       
       for m=1:n_neg %Recorre todos los VS negativos
           idx_vs_neg = indice_vs_neg(m);  
           Sum = 0;
               for i=1:num_vs, %Recorre todos los VS
                     idx_vs   = indice_vs(i);  
                     Sum = Sum + a(idx_vs)*Ytrain_res(idx_vs)*((1+Xtrain_res(idx_vs,:)-Xtrain_res(idx_vs_neg,:)')^paramKernel);
               end
         b_neg(m) = -1 - Sum;
       end
        
end    


b_opt_pos = mean(b_pos);
b_opt_neg = mean(b_neg);

b_opt = ((n_pos*b_opt_pos)+ (n_neg*b_opt_neg))/(n_pos+n_neg);

labels_svm=NaN(ntest,1);

switch lower(tipoKernel)
    case 'sin kernel'
        for i_test=1:ntest
            aux=0;
            for i=1:num_vs, 
                 idx_vs = indice_vs(i);  
                 aux = aux + a(idx_vs)*Ytrain_res(idx_vs)*(Xtrain_res(idx_vs,:)'*Xval(i_test,:));
            end
            labels_svm(i_test,1) = sign(aux);
        end
    case 'gaussiano'
        for i_test=1:ntest
            aux=0;
            for i=1:num_vs,
                 idx_vs = indice_vs(i);  
                 dif2=norm(Xtrain_res(idx_vs,:)-Xval(i_test,:));
                 aux = aux + a(idx_vs)*Ytrain_res(idx_vs)*(exp(-0.5*(dif2*dif2')/(paramKernel^2)));
            end
            labels_svm(i_test,1) = sign(aux);
        end
    case 'polinomico'
        for i_test=1:ntest
            aux=0;
            for i=1:num_vs,
                 idx_vs = indice_vs(i);  
                 aux = aux + a(idx_vs)*Ytrain_res(idx_vs)*((1+Xtrain_res(idx_vs,:)-Xval(i_test,:)')^paramKernel);
            end
            labels_svm(i_test,1) = sign(aux);
        end  
end       

clase_1=999999999;
clase_2=-999999999;
labels_svm(labels_svm==-1)=clase_1;
labels_svm(labels_svm==1)=clase_2;
labels_svm(labels_svm==clase_1)=clases(1);
labels_svm(labels_svm==clase_2)=clases(2);


end