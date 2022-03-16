 # -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:28:56 2021
Index: 18880323
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics 
import statsmodels.api as sm

# Reading the csv file and parsing the missing values
df = pd.read_csv('Adult_Final.csv', na_values=' ?')
print(df.head())

print('Shape of the dataset:',df.shape)
print('\n')

# Total missing values in the data
#print('Number of missing data before dropping noisy attributes')
#print(df.isnull().sum())
#print('\n')

# Dropping noisy attributes
df = df.drop(columns = ['fnlwgt','edu-num','capital-gain','capital-loss','hrs','country'])

print('Number of missing data after dropping noisy attributes')
print(df.isnull().sum())
print('\n')

# Removing rows with missing values
df.dropna(how='any',inplace=True)

print('After dropping missing values')
print('Shape of the dataset:',df.shape)
print('\n')

# Converting categorical data to numerical data
workclass = {' Private':1, ' Self-emp-not-inc':2, ' Local-gov':3, ' State-gov':4, ' Self-emp-inc':5, ' Federal-gov':6, ' Without-pay':7, ' Never-worked':8}
education = {' HS-grad':1, ' Some-college':2, ' Bachelors':3, ' Masters':4, ' Assoc-voc':5, ' 11th':6, ' Assoc-acdm':7, ' 10th':8, ' 7th-8th':9, ' Prof-school':10, ' 9th':11, ' 12th':12, ' Doctorate':13, ' 5th-6th':14, ' 1st-4th':15, ' Preschool':16}
marital = {' Married-civ-spouse':1, ' Never-married':2, ' Divorced':3, ' Separated':4, ' Widowed':5, ' Married-spouse-absent':6, ' Married-AF-spouse':7}
occupation = {' Prof-specialty':1, ' Craft-repair':2, ' Exec-managerial':3, ' Adm-clerical':4, ' Sales':5, ' Other-service':6, ' Machine-op-inspct':7, ' Transport-moving':8, ' Handlers-cleaners':9, ' Farming-fishing':10, ' Tech-support':11, ' Protective-serv':12, ' Priv-house-serv':13, ' Armed-Forces':14}
relationship = {' Husband':1, ' Not-in-family':2, ' Own-child':3, ' Unmarried':4, ' Wife':5, ' Other-relative':6}
race = {' White':1, ' Black':2, ' Asian-Pac-Islander':3, ' Amer-Indian-Eskimo':4, ' Other':5}
gender = {' Male':1, ' Female':2}
income = {' <=50K':0, ' >50K':1}

df.workclass = [workclass[item] for item in df.workclass]
df.education = [education[item] for item in df.education]
df.marital = [marital[item] for item in df.marital]
df.occupation = [occupation[item] for item in df.occupation]
df.relationship = [relationship[item] for item in df.relationship]
df.race = [race[item] for item in df.race]
df.gender = [gender[item] for item in df.gender]
df.income = [income[item] for item in df.income]

df = df.drop(df.query('income == 0').sample(frac=.60).index)

plt.figure()
#sns.distplot(a = df['age'], kde = False)
df.age.hist()
plt.title('Age Distribution')
plt.show()

# Binning the age values for 3 seperate groups
# 16-45, 46-70, >70
bins = [16,45,70,100]
labels = [1,2,3]
df['age'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

def graph(df,attr):
    plt.figure()
    df.groupby(attr).income.mean().plot(kind='bar')
    plt.show()

graph(df,'workclass')
graph(df,'education')
graph(df,'marital')
graph(df,'occupation')
graph(df,'relationship')
graph(df,'race')
graph(df,'gender')

print('Dataset after preprocessing')
print(df.head())
print('\n')
# print('Shape of the dataset after preprocessing:',df.shape)
# print('\n')
print('Income values counts')
print (df['income'].value_counts())
print('\n')

# Convert to numpy array and getting the X and Y data
data = df.to_numpy()
X = data[:,0:8]
X = X.astype('int')
Y = data[:,8]
Y = Y.astype('int')

""" Applying Logistic Regression and getting the metrics """
# Linear regression model and splitting the data into train(67%) and test set(33%)
reg = LogisticRegression()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

# Training the model 
reg.fit(x_train, y_train)

# Predictions on test data
y_pred = reg.predict(x_test)

# Summary on regression result and printing the summary table
logit_model=sm.Logit(Y,X).fit()
print('Summary table')
print(logit_model.summary())
print('\n')

# Generating confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
 
# Plot confusion matrix
plt.figure(figsize=(9,9))
sns.heatmap(cnf_matrix, annot=True, fmt='g', linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
plt.title('Confusion matrix', size = 15);
plt.show()
 
# Print the accuracy, precision and recall of the model
print('Accuracy: %.2f%%' % (metrics.accuracy_score(y_test, y_pred)*100))
print('Precision:%.2f%%' % (metrics.precision_score(y_test, y_pred)*100))
print('Recall:%.2f%%' % (metrics.recall_score(y_test, y_pred)*100))
print('\n')

# ROC curve
y_pred_proba = reg.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.title('ROC Curve')
plt.legend(loc=4)
plt.show()

# Applying cross validation 
kfold = KFold(n_splits=5, random_state=0)
model_kfold = LogisticRegression()
results_kfold = cross_val_score(model_kfold, X, Y, cv=kfold)
print("Accuracy after cross validation: %.2f%%" % (results_kfold.mean()*100.0)) 




