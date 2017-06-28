% function [y,x] = parzen(data,x,sigma);
% --------------------------------------------------
% Author: Jussi Tohka, Institute of Signal Processing
% Tampere University of Technology, Finland
% --------------------------------------------------
% Permission to use, copy, modify, and distribute this software 
% for any purpose and without fee is hereby
% granted.  The author and Tampere University of Technology 
% make no representations
% about the suitability of this software for any purpose.  It is
% provided "as is" without express or implied warranty

function [y,x] = parzen(data,varargin);
if length(varargin) > 0
     if length(varargin{1}) > 1
       x = varargin{1};
     else
       step = (max(data) - min(data))/varargin{1};
        x = min(data):step:max(data);
     end
else 
     step = (max(data) - min(data))/100;
     x = min(data):step:max(data);
end
if length(varargin) > 1
     sigma = varargin{2};
else
%     step = (max(data) - min(data))/100;   
     sigma = step;
end


y = zeros(size(x));
for i = 1:length(x)
  y(i) = sum(normpdf(data,x(i),sigma));
end
y = y/length(data);