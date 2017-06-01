
# Import required libraries and load data file

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split

fruits=pd.read_table("D:\\UM- Applied Machine Learning\\fruit_data_with_colors.txt")
fruits.head()

# create a mapping from fruit label value to fruit name 
lookup_fruit_name=dict(zip(fruits.fruit_label.unique(),fruits.fruit_name.unique()))
lookup_fruit_name

# Examining the data
# plotting a scatter matrix
from matplotlib import cm
X=fruits[['height','width','mass','color_score']]
y=fruits['fruit_label']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
cmap=cm.get_cmap('gnuplot')
scatter=pd.scatter_matrix(X_train,c=y_train,marker='o',s=40,hist_kwds={'bins':15},figsize=(9,9),cmap=cmap)

# plotting a 3d scatter plot
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(X_train['width'],X_train['height'],X_train['color_score'],c=y_train,marker='o',s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
plt.show()

# create train test split
X=fruits[['mass','width','height']]
y=fruits['fruit_label']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

# create classifier
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)

# train the classifier
knn.fit(X_train,y_train)

# estimate the accuracy of the model on the test data
knn.score(X_test,y_test)

# Use the knn model to predict new,previosly unseen data
fruit_prediction=knn.predict([[20,4.3,5.5]])
lookup_fruit_name[fruit_prediction[0]]

fruit_prediction=knn.predict([[100,6.3,8.5]])
lookup_fruit_name[fruit_prediction[0]]

# plot the decision boundaries of the k-nn classifier
# from adspy_shared_utilities import plot_fruit_knn

# how sensitive is the k-nn classification accuracy to the choice of the k parameter
k_range=range(1,20)
scores=[]

for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    scores.append(knn.score(X_test,y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range,scores)
plt.xticks([0,5,10,15,20]);

# how sensitive is k-nn to the train test split proportion
t=[0.8,0.7,0.6,0.5,0.4,0.3,0.2]
knn=KNeighborsClassifier(n_neighbors=5)

plt.figure()
for s in t:
    scores=[]
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy');















