function [] = plotKPCA(proy,testset,X)
x=sort(unique(testset(:,1))); nx = length(x);
y=sort(unique(testset(:,2))); ny = length(y);

for n = 1:size(proy,2),
    figure;
    axis([min(x) max(x) min(y) max(y)]);
    imag = reshape(proy(:,n), ny, nx);
    colormap(gray); 
    hold on;
    pcolor(x, y, imag);
    shading interp 
    contour(x, y, imag, 12, 'b');
    plot(X(:,1), X(:,2), 'r.')
end
