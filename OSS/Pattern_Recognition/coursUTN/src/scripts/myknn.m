function [Yhat]=myknn(Xts,Xtrain,Ytrain,K)
    Ntrain=length(Xtrain)
    Ntest=size(Xts,1)
    D=zeros(Ntrain,Ntest); I=zeros(K,Ntest), Yhat=zeros(Ntest,1)
    for j=1:Ntest
        for i=1:Ntrain
            D(i,j)=norm(Xts(j,:)-Xtrain(i,:));
        end
        [value,idx]=sort(D(:,j))
        Yhat=sign(sum(Ytrain(idx(1,K))))        
    end
end
