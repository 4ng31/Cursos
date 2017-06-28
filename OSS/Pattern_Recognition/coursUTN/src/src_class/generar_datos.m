function [Y,X] = generar_datos(N,caso)
if nargin < 2,
    caso = 2
end
    switch caso
        case 1,
            %the total number of points is 5*N
            X1=randn(N,2);
            X2=randn(N,2)+ones(N,1)*[ 4 4];
            X3=randn(N,2)+ones(N,1)*[ -4 4];
            X4=randn(N,2)+ones(N,1)*[ 4 -4];
            X5=randn(N,2)+ones(N,1)*[ -4 -4];
 
            Y1=ones(N,1);   %labels of class 1
            Y2=-ones(4*N,1);  %labels of class 2

            % creation of the data base
            X=[X1;X2;X3;X4;X5];
            Y=[Y1;Y2];
        case 2,
            % the total number of points is 2*N
            rho=10*(.5:2.5/N:3-2.5/N);
            theta=pi:4*pi/N:5*pi-4*pi/N;
            X1=[rho'.*cos(theta') rho'.*sin(theta')];
            X2=[-rho'.*cos(theta') -rho'.*sin(theta')];
            Y1=ones(N,1);   %labels of class 1
            Y2=-ones(N,1);  %labels of class 2
            % creation of the data base
            X=[X1;X2];
            Y=[Y1;Y2];
        case 3 
            d=2; %d : dimension of the data (here 2 to be represented graphically)
            X1=randn(N,d);    %generation of the fist class data set
            X2=1.5*randn(N,d)+1.75*ones(N,1)*(1*ones(1,d));    %generation of the fist class data set
            Y1=ones(N,1);   %labels of class 1
            Y2=-ones(N,1);  %labels of class 2
            X=[X1;X2];
            Y=[Y1;Y2];
        case 4
            delx=6/N;
            x=-3:delx:3;
            y1=sqrt(6-x.^2); y1=y1+.65*randn(size(y1));
            y2=-sqrt(6-x.^2);y2=y2+.65*randn(size(y2));
            X2=.4*randn(100,2);
            X1=[[x' y1'];[x' y2']];
            Y1=ones(length(X1),1);
            Y2=-ones(length(X2),1);
            X=[X1;X2];
            Y=[Y1;Y2]; 
    end
    gscatter(X(:,1),X(:,2),Y)