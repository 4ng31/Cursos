function [y]=mysvc(NET, xt, yt, x, alpha, b)
    ker = NET.ker;	   
    kerpar = NET.kerpar;
    y = kernel(ker, kerpar, x, xt)*(alpha.*yt) + b;
