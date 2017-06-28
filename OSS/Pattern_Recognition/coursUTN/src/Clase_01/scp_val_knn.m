clc;
clear all;
close all;

[Y,X] = generar_datos(1000,2);

k = 5 ;
X_train = X ;
Y_train = Y ;

Xts = [1 5 20]' ;


nTest = 10 ;
nTrain = 10 ;

n = length(Y) ;

error = 0 ;

for i=1:n
    
    Xts = X(i,:) ;
    Yts = X(i);
    
    X_train =  X ; % menos el i
    X_train(i,:) = [] ;
    
    Y_train =  Y ; % menos el i
    Y_train(i,:) = [] ;
    
    Y_hat = func_knn(Xts , X_train , Y_train , k ) ;
    
    if Y_hat ~= Y_ts 
        error = error + 1 ;
    end
    
end

Tasa = error / n ;
disp(sprintf('Tasa Error: %.2f',Tasa));


return 

[Y,X] = generar_datos(1000,2);

k = 5 ;
X_train = X ;
Y_train = Y ;

Xts = [1 5 20]' ;

[Y_hat] = func_knn(Xts , X_train , Y_train , k ) ;

Y_hat




