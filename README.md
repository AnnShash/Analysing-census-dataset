# Analysing-census-dataset
The goal is to predict whether a person earns more than 50k per year or not.  

Description. <br/>
Dataset used was adult.csv which contains 48842 observations and 14 features including 6 numerical features and 8 categorical features. 
Source: https://archive-beta.ics.uci.edu/ml/datasets/adult <br/>

Preprocessing. <br/>
Before reading the dataset the adult data and the test data were combined as a single data set and the column names were set manually (Adult_Final.csv). In the test dataset the column ‘income’ contained a ‘.’ at the end of each record. This was removed manually. In the dataset missing values were represented by ‘?’, therefore it was passed as missing values so that Python can identify it as a missing value. <br/>
After reading the data, the noisy attributes ('fnlwgt','edu-num','capital-gain','capital-loss','hrs','country') were dropped. After that the number of missing values were computed. <br/>
2799 data were missing from the workclass attribute and 2809 were missing from occupation attribute. After dropping the rows which contained missing values the final dataset consists of 46033 records, which means around 5.7% records contained missing information. <br/>
When analyzing the income attribute, it was found that 34611(75%) records have an income less than 50K and remaining have an income more than 50K. Therefore, the dataset is imbalanced.<br/><br/>
Next the categorical data were mapped to numerical values as follows<br/>
•	workclass = {' Private':1, ' Self-emp-not-inc':2, ' Local-gov':3, ' State-gov':4, ' Self-emp-inc':5, ' Federal-gov':6, ' Without-pay':7, ' Never-worked':8}<br/>
•	education = {' HS-grad':1, ' Some-college':2, ' Bachelors':3, ' Masters':4, ' Assoc-voc':5, ' 11th':6, ' Assoc-acdm':7, ' 10th':8, ' 7th-8th':9, ' Prof-school':10, ' 9th':11, ' 12th':12, ' Doctorate':13, ' 5th-6th':14, ' 1st-4th':15, ' Preschool':16}<br/>
•	marital = {' Married-civ-spouse':1, ' Never-married':2, ' Divorced':3, ' Separated':4, ' Widowed':5, ' Married-spouse-absent':6, ' Married-AF-spouse':7}<br/>
•	occupation = {' Prof-specialty':1, ' Craft-repair':2, ' Exec-managerial':3, ' Adm-clerical':4, ' Sales':5, ' Other-service':6, ' Machine-op-inspct':7, ' Transport-moving':8, ' Handlers-cleaners':9, ' Farming-fishing':10, ' Tech-support':11, ' Protective-serv':12, ' Priv-house-serv':13, ' Armed-Forces':14}<br/>
•	relationship = {' Husband':1, ' Not-in-family':2, ' Own-child':3, ' Unmarried':4, ' Wife':5, ' Other-relative':6}<br/>
•	race = {' White':1, ' Black':2, ' Asian-Pac-Islander':3, ' Amer-Indian-Eskimo':4, ' Other':5}<br/>
•	gender = {' Male':1, ' Female':2}<br/>
•	income = {' <=50K':0, ' >50K':1}<br/><br/>

Model<br/>
The objective is to predict whether the income is greater than 50K or not, therefore the Logistic regression method was used. The data were split into train set (67%) and test set (33%) using the train_test_split() function. The model was trained using .fit() function and the outputs were predicted using .predict() function.</br><br/>

Improvements <br/>
•	The created model can be improved by using a balanced dataset.
•	Accuracy can be improved by adding some more significant attributes.
•	Can create a more accurate model by using different classifiers such as Support Vector Machines and Decision Tress.



