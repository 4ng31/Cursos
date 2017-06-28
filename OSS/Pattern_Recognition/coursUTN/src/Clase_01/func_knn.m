
function [Y_hat] = func_knn(Xts , X_train , Y_train , k )

nTrain = length(Y_train);

nTest  = size(Xts,1);

D = zeros(nTrain,nTest);

I = zeros(k,nTest) ;


    for j=1:nTest
        
        for i=1:nTrain      
            D(i,j) = norm(Xts(j,:)-X_train(j,:));
        end
    
        [ Val , Pos ] = sort( D(:,j) ) ;
    
        I(:,j) = Pos(1:k) ;
    
        Y_hat_1 =  Y_train(I(:,j)) ;

        Y_hat(j) = sign ( sum( Y_hat_1 ) );
        
    end







return;




[ D , I ] = pdist2(X,Xts,'euclidean','smallest',5);

for n=1:length(labels),
    labels(n)=sign(sum(Y(I(:,n))));
end




ngrid = 1000;
[XX,YY] = meshgrid(linspace(-30,30,ngrid),linspace(-30,30,ngrid));

z1=XX(:); z2=YY(:);
Xts=[z1 z2];



labels=zeros(length(Xts),1);



