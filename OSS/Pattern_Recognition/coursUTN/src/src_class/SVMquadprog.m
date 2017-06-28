function [alpha] = SVMquadprog(NET,X,Y)
if size(X,1)~=length(Y),
    error('X e Y son incompatibles');
end
N = length(Y);
C=NET.C;                % parameter C
kerneltype=NET.ker;     % kernel type
kernelpar=NET.kerpar;   % kernel parameter

disp('quadprog')
H=(Y*Y').*kernelmtx(kerneltype,kernelpar,X,X);
f=-ones(N,1);
A=[];b=[];
Aeq=Y';
beq=0;
LB=zeros(N,1); UB=C*ones(N,1);
X0=zeros(N,1);
MaxIter=100000*N;
options=optimset('quadprog');
options=optimset(options,'MaxIter',MaxIter);
alpha=quadprog(H,f,A,b,Aeq,beq,LB,UB,X0, options);

