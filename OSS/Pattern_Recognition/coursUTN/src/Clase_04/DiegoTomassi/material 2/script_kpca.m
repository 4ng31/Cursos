clear all; close all;
load patterns.mat

x = linspace(min(patterns(:,1)),max(patterns(:,1)),40);
y = linspace(min(patterns(:,2)),max(patterns(:,2)),40);
[X,Y]=meshgrid(x,y);
testset = [X(:) Y(:)];

proyKPCA = kpca(patterns,'rbf',.1,6,testset);
disp('Graficando royecciones de KPCA...')
plotKPCA(proyKPCA,testset,patterns);

pause;
% LINEAR PCA
Xc = testset - repmat(mean(testset),size(testset,1),1);
N=size(patterns,1);
sigma = cov(patterns)*(N-1)/N;
[evec,eval] = eig(sigma);
[eval,idx] = sort(diag(eval),'descend'); evec = evec(:,idx);
proyPCA = Xc*evec + repmat(mean(testset),size(testset,1),1);
disp('Graficando royecciones de PCA...')
plotKPCA(proyPCA,testset,patterns);

pause;


% vistos como clases distintas
unos = ones(30,1);
labels = [unos; 2*unos; 3*unos];
figure;
gscatter(patterns(:,1),patterns(:,2),labels);

pause;
disp('Graficando proyecciones etiquetadas de KPCA...')
figure;
proyKPCA = kpca(patterns,'rbf',.1,6,patterns);
gscatter(proyKPCA(:,1),proyKPCA(:,1),labels);

pause;
disp('Graficando proyecciones etiquetadas de PCA...')
figure;
proyPCA = (patterns - repmat(mean(patterns),N,1))*evec + repmat(mean(patterns),N,1);
gscatter(proyPCA(:,2),proyPCA(:,2),labels);

