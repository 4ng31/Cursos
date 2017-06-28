function Kxy = kernel(kertype, kerparam,x,y)
switch lower(kertype)
    case 'poly',
        Kxy = (x*y'+1)^kerparam;
    case 'rbf',
        er = x-y;
        Kxy = exp(-er*er'/(2*kerparam*kerparam));
end