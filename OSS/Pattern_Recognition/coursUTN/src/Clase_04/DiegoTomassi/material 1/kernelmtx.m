function [K] = kernelmtx(kerneltype,kernelpar,X1,X2)
[n1]=size(X1,1);
[n2]=size(X2,1);

K=zeros(n1,n2);
for i=1:n1,
    for j=1:n2,
        K(i,j) = kernelpt(kerneltype,kernelpar,X1(i,:),X2(j,:));
    end
end