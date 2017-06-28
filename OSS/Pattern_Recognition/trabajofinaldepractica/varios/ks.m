function yhat = ks(x,y,opt,h)

% ks  local constant/local linear kernel regression

% x: n-by-k matrix
% y: n-by-1 vector
% h: a positive scalar
% opt: 'kr' or [] if (local constant) kernel regression, 'krll' if local linear kernel regression

% yhat = ks(x,y,opt,h) implements a multi-dimensional nonparametric regression, 
% i.e., kernel regression (the default) or local linear regression using the Quartic kernel. 
% yhat = ks(x,y,opt,h) returns the fitted value yhat at each obs. of x.

% yhat = ks(x,y) performs kernel regression using the optimal bandwidth estimated by cross validation (calls function opt_h).
% yhat = ks(x,y,opt) performs the specified (by opt) nonparametric regression using the optimal bandwidth estimated by cross validation.
% yhat = ks(x,y,opt,h) performs the specified (by opt) nonparametric regression using the provided bandwidth, h.

% Copyright: Yingying Dong at Boston College, July, 2008.

%%

error(nargchk(2,4,nargin));
error(nargchk(1,1,nargout));

if nargin < 3
    opt = [];
elseif ~(strcmp(opt, 'kr')||strcmp(opt,'krll')||isempty(opt))
    error('opt must be ''kr'', ''krll'' or [].')
end

if nargin < 4
    h =  opt_h(x,y,opt);
elseif ~isscalar(h)
    error('The provided bandwidth must be a scalar.')
end

[n k]= size(x); b=zeros(k,n); yhat = zeros(n,1);

for i = 1:n
    dis = bsxfun(@minus, x, x(i,:));
    u = bsxfun(@rdivide, dis, std(dis))/h;
    Kernel = @(u)15/16*(1-u.^2).^2.*(abs(u)<=1);
    w = prod(Kernel(u),2);
    sw = sum(w);
    if strcmp(opt, 'krll')
        t1 = bsxfun(@times, dis, w)';
        t2 = sum(t1,2);
        b(:,i)= (sw*t1*dis - t2*t2')\(sw*t1*y - t2*w'*y);
        yhat(i) = w'*(y - dis*b(:,i))/sw;
    else
        yhat(i) = w'*y/sw;
    end
end

fprintf('\n%4.0f fitted values are generated  \n', n);


