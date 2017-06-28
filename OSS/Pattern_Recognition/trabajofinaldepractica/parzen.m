clc
load '/home/bgx/trabajofinaldeprctica/datosOS14.mat'

datos.X;
X=datos.X;
Y=datos.Y;
nRows=size(X,1);
randRows=randperm(nRows);%# generate random ordering of row indices
Xtrain=X(randRows(1:80),:);
Xtest=X(randRows(81:end),:);
Ytrain=Y(randRows(1:80),:);
Ytest=Y(randRows(81:end),:);
sigma=1

NumFeatures = size(Xtrain,2);
NumPatternTrain = size(Xtrain,1);
NumPatternToTest = size(Xtest,1);

Yhat = NaN * zeros(NumPatternToTest,1);

Mat_Sigma = sigma * eye(NumFeatures);
%Det_Sigma = det(Mat_Sigma);
%Inv_Sigma = inv(Mat_Sigma);

  for i_test = 1 : NumPatternToTest
        %Gauss=mvnpdf(X,MU,SIGMA);
        Gauss=mvnpdf( Xtest(:,i_test) , Xtrain, Mat_Sigma ) ;
        Acum = Ytrain * Gauss ;
        Yhat(i_test) =  sign(Acum);
  end