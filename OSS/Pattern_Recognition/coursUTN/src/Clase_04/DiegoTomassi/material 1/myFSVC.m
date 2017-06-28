function[gamma]=myFSVC(NET,X,Y)

% INPUTS
% NET : structure defining the training parameters (help createnet)
% X : training data matrix (N observations of size d)
% Y : column vector containing labels (-1, +1) of the training set
% 

%OUTPUTS
% gamma : lagrange multipliers obtained after the optimization process. 
% To come back to usual alpha : alpha=gamma.*Y;
% b: bias

C=NET.C;                % parameter C
kerneltype=NET.ker;     % kernel type
kernelpar=NET.kerpar;   % kernel parameter

[N,d]=size(X);    % size of the training data set



gamma=zeros(N,1); % initialization of gamma = alpha.*Y
g=Y;%% initialization of the gradient


criterio=1; % initialisation of the stopping criterion

LB=min(0,C*Y); % bounds defining the feasible domain
UB=max(0,C*Y);
k=0;    

while criterio>2*NET.kktol; % loop until convergence
    k=k+1;
    contrA=find(gamma>LB);
    contrB=find(gamma<UB);
    [aux,indi] = max(g(contrB)); indi = contrB(indi);
    [aux,indj] = min(g(contrA)); indj = contrA(indj);

    criterio=abs(g(indi)-g(indj)); %convergence criterion
% 
    kii=kernel(kerneltype,kernelpar,X(indi,:),X(indi,:));% kernel between (i,i)
    kjj=kernel(kerneltype,kernelpar,X(indj,:),X(indj,:));% kernel between (j,j)
    kij=kernel(kerneltype,kernelpar,X(indi,:),X(indj,:));% kernel between (i,j)
    K1=kii+kjj-2*kij;% needed for the optimal step size calculation


    lambda=(g(indi)-g(indj))/K1; % determination of the optimal step size

    kern1=kernel(kerneltype,kernelpar,X(indi,:),X)-kernel(kerneltype,kernelpar,X(indj,:),X); % to be used later

    % Clipping of the optimal step size to stay into the feasible domain
    maxlambda=min(UB(indi)-gamma(indi),gamma(indj)-LB(indj));
    minlambda=max(LB(indi)-gamma(indi),gamma(indj)-UB(indj));
    if lambda > maxlambda 
        lambda=maxlambda;
    elseif lambda<minlambda
        lambda=minlambda;
    end;

    % update of the solution
    gamma(indi)=gamma(indi)+lambda;
    gamma(indj)=gamma(indj)-lambda;

    %update of the gradient
    g=g-lambda*kern1';

end;



