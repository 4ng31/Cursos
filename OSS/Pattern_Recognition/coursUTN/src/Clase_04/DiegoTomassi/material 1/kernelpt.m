function kij = kernelpt(kerneltype,kernelpar,x,y)
% evalua el kernel k(.,.) en los puntos x e y (k(x,y))
switch lower(kerneltype)
    case 'poly'
        kij = (1+x*y')^kernelpar;
    case 'rbf'
        er=x-y;
        kij = exp(-.5*(er*er')/(kernelpar^2));
    case 'idem'
	kij = x*y';
end