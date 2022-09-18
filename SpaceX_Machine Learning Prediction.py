#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDS0321ENSkillsNetwork26802033-2022-01-01" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# # **Space X  Falcon 9 First Stage Landing Prediction**
# 

# ## Assignment:  Machine Learning Prediction
# 

# Estimated time needed: **60** minutes
# 

# Space X advertises Falcon 9 rocket launches on its website with a cost of 62 million dollars; other providers cost upward of 165 million dollars each, much of the savings is because Space X can reuse the first stage. Therefore if we can determine if the first stage will land, we can determine the cost of a launch. This information can be used if an alternate company wants to bid against space X for a rocket launch.   In this lab, you will create a machine learning pipeline  to predict if the first stage will land given the data from the preceding labs.
# 

# ![](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/Images/landing\_1.gif)
# 

# Several examples of an unsuccessful landing are shown here:
# 

# ![](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/Images/crash.gif)
# 

# Most unsuccessful landings are planed. Space X; performs a controlled landing in the oceans.
# 

# ## Objectives
# 

# Perform exploratory  Data Analysis and determine Training Labels
# 
# *   create a column for the class
# *   Standardize the data
# *   Split into training data and test data
# 
# \-Find best Hyperparameter for SVM, Classification Trees and Logistic Regression
# 
# *   Find the method performs best using test data
# 

# 

# ***
# 

# ## Import Libraries and Define Auxiliary Functions
# 

# We will import the following libraries for the lab
# 

# In[344]:


# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier


# This function is to plot the confusion matrix.
# 

# In[345]:


def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['did not land', 'land']); ax.yaxis.set_ticklabels(['did not land', 'landed'])


# ## Load the dataframe
# 

# Load the data
# 

# In[346]:


#data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv")

# If you were unable to complete the previous lab correctly you can uncomment and load this csv

data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/dataset_part_2.csv')

data.head()


# In[347]:


#X = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_3.csv')

# If you were unable to complete the previous lab correctly you can uncomment and load this csv

X = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/dataset_part_3.csv')

X.head(100)


# ## TASK  1
# 

# Create a NumPy array from the column <code>Class</code> in <code>data</code>, by applying the method <code>to_numpy()</code>  then
# assign it  to the variable <code>Y</code>,make sure the output is a  Pandas series (only one bracket df\['name of  column']).
# 

# In[348]:


Y=data['Class'].to_numpy()


# In[349]:


type(Y)


# ## TASK  2
# 

# Standardize the data in <code>X</code> then reassign it to the variable  <code>X</code> using the transform provided below.
# 

# In[350]:


# students get this 
transform = preprocessing.StandardScaler()


# In[351]:


X = transform.fit(X).transform(X)


# In[352]:


X


# We split the data into training and testing data using the  function  <code>train_test_split</code>.   The training data is divided into validation data, a second set used for training  data; then the models are trained and hyperparameters are selected using the function <code>GridSearchCV</code>.
# 

# ## TASK  3
# 

# Use the function train_test_split to split the data X and Y into training and test data. Set the parameter test_size to  0.2 and random_state to 2. The training data and test data should be assigned to the following labels.
# 

# <code>X_train, X_test, Y_train, Y_test</code>
# 

# In[353]:


X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=2)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)


# we can see we only have 18 test samples.
# 

# In[354]:


Y_test.shape


# ## TASK  4
# 

# Create a logistic regression object  then create a  GridSearchCV object  <code>logreg_cv</code> with cv = 10.  Fit the object to find the best parameters from the dictionary <code>parameters</code>.
# 

# In[355]:


parameters ={'C':[0.01,0.1,1],
             'penalty':['l2'],
             'solver':['lbfgs']}


# In[356]:


parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge
lr=LogisticRegression()
#LR = LogisticRegression(C=0.01, solver='liblinear')


# In[357]:


logreg_cv=GridSearchCV(estimator=lr, param_grid=parameters, scoring='accuracy', cv=10)
logreg_cv.fit(X,Y)
logreg_cv


# In[358]:


logreg_cv.cv_results_


# We output the <code>GridSearchCV</code> object for logistic regression. We display the best parameters using the data attribute <code>best_params\_</code> and the accuracy on the validation data using the data attribute <code>best_score\_</code>.
# 

# In[359]:


print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


# ## TASK  5
# 

# Calculate the accuracy on the test data using the method <code>score</code>:
# 

# In[360]:


logreg_cv.score(X_test,Y_test)
#print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))


# Lets look at the confusion matrix:
# 

# In[361]:


yhat=logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# Examining the confusion matrix, we see that logistic regression can distinguish between the different classes.  We see that the major problem is false positives.
# 

# ## TASK  6
# 

# Create a support vector machine object then  create a  <code>GridSearchCV</code> object  <code>svm_cv</code> with cv - 10.  Fit the object to find the best parameters from the dictionary <code>parameters</code>.
# 

# In[362]:


parameters = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()


# In[363]:


svm_cv=GridSearchCV(svm, parameters,scoring='accuracy',cv=10)
svm_cv.fit(X,Y)
svm_cv


# In[364]:


print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)


# ## TASK  7
# 

# Calculate the accuracy on the test data using the method <code>score</code>:
# 

# In[365]:


svm_cv.score(X_test,Y_test)


# We can plot the confusion matrix
# 

# In[366]:


yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# ## TASK  8
# 

# Create a decision tree classifier object then  create a  <code>GridSearchCV</code> object  <code>tree_cv</code> with cv = 10.  Fit the object to find the best parameters from the dictionary <code>parameters</code>.
# 

# In[367]:


parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()


# In[368]:


tree_cv=GridSearchCV(estimator=tree, param_grid=parameters, cv=10)
tree_cv.fit(X,Y)


# In[369]:


print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)


# ## TASK  9
# 

# Calculate the accuracy of tree_cv on the test data using the method <code>score</code>:
# 

# In[370]:


tree_cv.score(X_test, Y_test)


# We can plot the confusion matrix
# 

# In[371]:


yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# ## TASK  10
# 

# Create a k nearest neighbors object then  create a  <code>GridSearchCV</code> object  <code>knn_cv</code> with cv = 10.  Fit the object to find the best parameters from the dictionary <code>parameters</code>.
# 

# In[372]:


parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
              'p': [1,2]}

KNN = KNeighborsClassifier()


# In[373]:


KNN_cv=GridSearchCV(estimator=KNN, param_grid=parameters,scoring='accuracy', cv=10)
KNN_cv.fit(X,Y)


# In[374]:


print("tuned hpyerparameters :(best parameters) ",KNN_cv.best_params_)
print("accuracy :",KNN_cv.best_score_)


# ## TASK  11
# 

# Calculate the accuracy of tree_cv on the test data using the method <code>score</code>:
# 

# In[375]:


KNN_cv.score(X_test, Y_test)


# We can plot the confusion matrix
# 

# In[376]:


yhat = KNN_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# ## TASK  12
# 

# Find the method performs best:
# 

# In[377]:



print('Logistic Regression Model: ',logreg_cv.score(X_test,Y_test))
print('Tree Model: ' ,tree_cv.score(X_test,Y_test))
print ('SVM Model: ' ,svm_cv.score(X_test,Y_test))
print ('KNN: ', KNN_cv.score(X_test,Y_test))

lg_test_score=logreg_cv.score(X_test,Y_test)
tree_test_score=tree_cv.score(X_test,Y_test)
svm_test_score=svm_cv.score(X_test,Y_test)
knn_test_score=KNN_cv.score(X_test,Y_test)


# In[378]:


algorithms = {'KNN':knn_test_score,'Tree':tree_test_score,'LogisticRegression':lg_test_score,
             'SVM':svm_test_score}
bestalgorithm = max(algorithms, key=algorithms.get)
print('Best Algorithm is',bestalgorithm,'with a score of',algorithms[bestalgorithm])
if bestalgorithm == 'Tree':
    print('Best Params is :',tree_cv.best_params_)
if bestalgorithm == 'KNN':
    print('Best Params is :',KNN_cv.best_params_)
if bestalgorithm == 'LogisticRegression':
    print('Best Params is :',logreg_cv.best_params_)
else:
    print('Best Params is :',svm_cv.best_params_)


# In[379]:


algorithms = {'KNN':KNN_cv.best_score_,'Tree':tree_cv.best_score_,'LogisticRegression':logreg_cv.best_score_,
             'SVM':svm_cv.best_score_}
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(algorithms.keys(), algorithms.values(), color ='grey',
        width = 0.4)
plt.ylim(0,1) 
plt.xlabel("Models")
plt.ylabel("Test Set Accuracy Score")
plt.title("Model Comparison on Training Data")
plt.show()


# In[380]:


algorithms = {'KNN':knn_test_score,'Tree':tree_test_score,'LogisticRegression':lg_test_score,
             'SVM':svm_test_score}
courses = list(algorithms.keys())
values = list(algorithms.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(algorithms.keys(), algorithms.values(), color ='blue',
        width = 0.4)
plt.ylim(0,1) 
plt.xlabel("Models")
plt.ylabel("Test Set Accuracy Score")
plt.title("Model Comparison on Test Data")
plt.show()


# In[ ]:





# ## Authors
# 

# <a href="https://www.linkedin.com/in/joseph-s-50398b136/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDS0321ENSkillsNetwork26802033-2022-01-01">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.
# 

# ## Change Log
# 

# | Date (YYYY-MM-DD) | Version | Changed By    | Change Description      |
# | ----------------- | ------- | ------------- | ----------------------- |
# | 2021-08-31        | 1.1     | Lakshmi Holla | Modified markdown       |
# | 2020-09-20        | 1.0     | Joseph        | Modified Multiple Areas |
# 

# Copyright © 2020 IBM Corporation. All rights reserved.
# 
