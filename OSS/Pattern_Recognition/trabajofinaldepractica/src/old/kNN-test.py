import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.cross_validation import train_test_split
  
# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
h = .01
# Create color maps from a list of colors
light_colors = ListedColormap(['blue', 'c', 'g'])
bold_colors  =  ListedColormap(['r', 'k', 'yellow'])
  
# uniform and distance are two arguments
for n_neighbors in [3,7]:
    for distancemetric in [1,2]:
        for algorithms in ['ball_tree', 'kd_tree', 'brute']:
            for weights in ['uniform', 'distance']:
                if (distancemetric == 1):
                    d_metric="Manhattan distance"
                else:
                    d_metric="Euclidean distance"
  
                clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights,algorithm=algorithms,p=distancemetric )
                clf.fit(X, y)
  
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                accuracy=clf.score(X,y)
                print("No of neighbors: "+str(n_neighbors)+", Distance metric: "+d_metric+", Algorithm is: " + algorithms +  ", weights: "+ weights+ ", Accuracy is: "+ str(accuracy))
      
                Z = Z.reshape(xx.shape)
                pl.figure()
                pl.pcolormesh(xx, yy, Z, cmap=light_colors )
                # Plot also the data points
                pl.scatter(X[:, 0], X[:, 1], c=y, cmap=bold_colors)
                pl.title("3-Class classification (k = %i, weights = '%s', algorithms ='%s',distance_metric= '%s')"  % (n_neighbors, weights,algorithms,d_metric))
                pl.axis('tight')               
pl.show()
