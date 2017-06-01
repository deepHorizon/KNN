
###### ASSIGNMENT 1 ##########
# breast cancer diagnosis

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()
cancer

# Print the dataset description
cancer.DESCR
cancer.keys()

# Question 0
# how many features does the breast cancer dataset have
def answer_zero():
    return len(cancer['feature_names'])
answer_zero()

# Question 1

def answer_one():
    global cancer
    target=pd.DataFrame(cancer.target,columns=['target'],index=pd.RangeIndex(start=0,stop=569,step=1))
    cancer=pd.DataFrame(cancer.data,columns=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
'mean smoothness', 'mean compactness', 'mean concavity',
'mean concave points', 'mean symmetry', 'mean fractal dimension',
'radius error', 'texture error', 'perimeter error', 'area error',
'smoothness error', 'compactness error', 'concavity error',
'concave points error', 'symmetry error', 'fractal dimension error',
'worst radius', 'worst texture', 'worst perimeter', 'worst area',
'worst smoothness', 'worst compactness', 'worst concavity',
'worst concave points', 'worst symmetry', 'worst fractal dimension',
],index=pd.RangeIndex(start=0,stop=569,step=1))
    cancer=pd.concat([cancer,target],axis=1)
    return cancer  
answer_one()

# Question 2

def answer_two():
    target=cancer.target
    target=target.map({0:'benign',1:'malignant'})
    target=target.groupby(target).count()
    return target
answer_two()
    
target=cancer.target.groupby(cancer.target).count()
target.index.map({0:'benign',1:'malignant'})
target=cancer.target.unique()
target=target.groupby(cancer.target).count()
cancer.target.groupby(cancer.target).count()
pd.Series(cancer.target.groupby(cancer.target).count(),index=['benign','malignant'])


def answer_three():
    X=cancer.drop(['target'],axis=1)
    y=cancer['target']
    
    
    
    return X, y


from sklearn.cross_validation import train_test_split

def answer_four():
    X, y = answer_three()
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
    # Your code here
    
    return X_train, X_test, y_train, y_test
  
  
from sklearn.neighbors import KNeighborsClassifier

def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    knn=KNeighborsClassifier(n_neighbors=1)
    
    # Your code here
    
    return knn.fit(X_train,y_train)  # Return your answer


def answer_six():
    means = cancer.mean()[:-1].values.reshape(1, -1)
    pred=knn.predict(means)
    # Your code here
    
    return pred# Return your answer

def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn=answer_five()
    pred=knn.predict(X_test)
    return pred
answer_seven()    

def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    score=knn.score(X_test,y_test)
    # Your code here
    
    return score# Return your answer
    

###PLOT
def accuracy_plot():
    import matplotlib.pyplot as plt

    %matplotlib notebook

    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)











