function [proyKPCA,v,lambda] = kpca(X,kertype,kerparam,d,testset)
if nargin < 5,
    testset = X;
end
[n,p] = size(X);
d = min(d,n);

% Armamos la GRAM MATRIX
K = zeros(n);
for i=1:n,
    for j=1:i,
        K(i,j) = kernel(kertype,kerparam,X(i,:),X(j,:));
        K(j,i) = K(i,j);
    end
end

% Centramos la GRAM MATRIX
unos = ones(n)/n;
Kc = K-unos*K-K*unos+unos*K*unos;

% Encontramos las direcciones principales
[a,lambda] = eig(Kc);
lambda = diag(lambda);
[lambda,idx] = sort(lambda,'descend');
a = a(:,idx);
v = a(:,1:d);
for j=1:d,
    v(:,j) = a(:,j)/sqrt(lambda(j));
end

        

% proyectamos los datos del testset
% Armamos la GRAM MATRIX
[ntest,p] = size(testset);
Ktest = zeros(ntest,n);
for i=1:ntest,
    for j=1:n,
        Ktest(i,j) = kernel(kertype,kerparam,testset(i,:),X(j,:));
    end
end
unos_test = ones(ntest,n)/n;

% K_test_n = K_test - unit_test*K - K_test*unit + unit_test*K*unit; 

Ktestc = Ktest - unos_test*K - Ktest*unos + unos_test*K*unos;
proyKPCA = Ktestc*v;

