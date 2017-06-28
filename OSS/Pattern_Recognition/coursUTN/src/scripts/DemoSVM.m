close all; clear all;

load NETparameters.mat;

N = 150;
[Y,X] = generar_datos(N,2);

tic;
alphaq = SVMquadprog(NET,X,Y);
tquad = toc;
bq = get_b(NET,alphaq,X,Y);
svcplot(NET,X,Y,alphaq,bq);
title('SVM usando QUADPROG')

tic;
gamma = myFSVC(NET,X,Y);
tfsvc = toc;
alpha= gamma.*Y;
b = get_b(NET,alpha,X,Y);
svcplot(NET,X,Y,alpha,b);
title('SVM usando F-SVC')

disp(tquad)
disp(tfsvc)