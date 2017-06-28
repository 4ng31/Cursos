clc;
clear all;
close all;


% T_FOLD_CrossVal

[Y,X] = generar_datos(1000,2);

n = length(Y) ;
idx = nperm(n);

Y=Y(idx) ;
X=X(idx,:) ;

% cuantos puntos hay en cada particion
aux = rand(N/t);  



for i=1:t
    
    idx = ((t-1)*aux + 1) : t*aux ;
    
    Yts = Y(idx) ;
    Xts = X(idx,:) ;
    
    Y_train = Y ;
    Y_train(idx) = [] ;

    
    k = 5 ;
    Y_hat = func_knn(Xts , X_train , Y_train , k ) ;
    % Y_hat tiene todos las respuetas a los elementos de lña particion

    Error_Part(i) = find(Y_hat ~= Yts) / length(Yts) ;
    
    
end

TasaError = mean(Error_Part)


return 

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




