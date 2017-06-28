xi=rand(1,1024);
x=linspace(-1,2,1024);
p=parzen1d(xi,x,0.5,[]);
figure,plot(x,p);
 
xi=randn(1,1024);
x=linspace(-2,2,1024);
p=parzen1d(xi,x,0.5,[]);
figure,plot(x,p);

clc,clear;

Num=10;
xi=mvnrnd([0 1 ],eye(2),Num)';
xi(2,:)=[];
x=linspace(-4,4,1024);
h1=0.5;
p=parzen1d(xi,x,h1,[]);
figure,plot(x,p);
title('parzen h1=0.5');
ylabel('N=10');
