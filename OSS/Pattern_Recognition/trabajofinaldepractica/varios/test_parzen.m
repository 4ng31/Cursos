% Parzen windows classifier applied to Fisher's iris data
% --------------------------------------------------
% Author: 2006 Jussi Tohka, Institute of Signal Processing
% Tampere University of Technology, Finland
% --------------------------------------------------
% Permission to use, copy, modify, and distribute this software 
% for any purpose and without fee is hereby
% granted. The author and Tampere University of Technology 
% make no representations
% about the suitability of this software for any purpose.  It is
% provided "as is" without express or implied warranty

load fisheriris
gscatter(meas(:,1),meas(:,2),species)
title('Training samples')
[y,x1] = parzen(meas(:,1),100,4*4/100);
[y,x2] = parzen(meas(:,2),100,4*4/100);
% Parzen estimation assuming features 1 and 2 independent
y1c1 = parzen(meas(1:50,1),x1,4*4/100); 
y1c2 = parzen(meas(51:100,1),x1,4*4/100); 
y1c3 = parzen(meas(101:150,1),x1,4*4/100); 
y2c1 = parzen(meas(1:50,2),x2,4*4/100); 
y2c2 = parzen(meas(51:100,2),x2,4*4/100); 
y2c3 = parzen(meas(101:150,2),x2,4*4/100); 

yc1 = kron(y2c1',y1c1);
yc2 = kron(y2c2',y1c2);
yc3 = kron(y2c3',y1c3);

figure
surf(x1,x2,yc1,3*ones(size(yc1)))
hold
surf(x1,x2,yc2,2*ones(size(yc1)))
surf(x1,x2,yc3,1*ones(size(yc1)))
title('Parzen estimates');
figure
surf(x1,x2,yc1,3*ones(size(yc1)))
hold
surf(x1,x2,yc2,2*ones(size(yc1)))
surf(x1,x2,yc3,1*ones(size(yc1)))
view(0,90)
title('Decision regions')