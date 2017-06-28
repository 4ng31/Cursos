function [Y_test] = func_MLR_Cuadratic(X_train, Y_train , X_test, label_C0,label_C1)

    [ NumPatternToTest ] = size(X_test,2);

    index_X_C0 = Y_train == label_C0 ;
    index_X_C1 = Y_train == label_C1 ;
    

    X_C0 = X_train(:,index_X_C0);
    X_C1 = X_train(:,index_X_C1);

    M_C0 = mean(X_C0,2);
    M_C1 = mean(X_C1,2);

    
    Cov_C0 =  inv(cov(X_C0'));
    Cov_C1 =  inv(cov(X_C1'));
    
    Cov_inv_C0 =  inv(Cov_C0);
    Cov_inv_C1 =  inv(Cov_C1);

        
    %U = + 0.5 * M_C0' * Cov_inv_C0 * M_C0 +...
    %    - 0.5 * M_C1' * Cov_inv_C1 * M_C1 +...
    %      0.5 * log10(det(Cov_C0)/det(Cov_C1)) ;

    U = + 0.5 * ( M_C0' / Cov_inv_C0 ) * M_C0 +...
        - 0.5 * ( M_C1' / Cov_inv_C1 ) * M_C1 +...
          0.5 * log10(det(Cov_C0)/det(Cov_C1)) ;
      
      
    %V_0 = - M_C0' * Cov_inv_C0 ;
    %V_1 = - M_C1' * Cov_inv_C1 ;
    
    V_0 = + M_C0' / Cov_C0 ;
    V_1 = + M_C1' / Cov_C1 ;
    
    
    T = 0.5 * (Cov_inv_C0 - Cov_inv_C1 );
    
    Y_test = zeros(1,NumPatternToTest);
    

    for i=1:NumPatternToTest
        
        X_act = X_test(:,i);
        
        Test_level = + X_act' * T * X_act +...
                     - V_0 * X_act +...
                     + V_1 * X_act +...
                     + U ;
        
        if Test_level < 0
            Y_test(i) = label_C0;
        else
            Y_test(i) = label_C1;
        end
        
    end;


end


