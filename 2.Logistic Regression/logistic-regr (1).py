# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 08:44:31 2022
Logistic regression
dataset: bank churn
"""

# import libraries
import pandas as pd
from numpy import unique
from sklearn.model_selection import train_test_split,KFold
import statsmodels.api as sm # Logit model
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import f_classif as fs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics # for ROC / AUC

# read the data
filename = "F:/aegis/4 ml/dataset/supervised/classification/bank/churn/bankchurn.csv"
churn = pd.read_csv(filename)

churn.head(3)
churn.shape
churn.columns

# remove features that are not reqd
churn.drop(columns=['custid','surname'],inplace=True)
churn.columns

# data types
churn.dtypes

# NULLs (on the entire dataset)
churn.isnull().sum()

# analysis based on the data types
# numeric cols
nc = churn.select_dtypes(exclude='object').columns.values 

# categorical cols
cc = churn.select_dtypes(include='object').columns.values
cc

# for the numeric EDA, copy the code from linear-regression

# to get the numeric data from the dataset
churn[nc]

# analyze the categorical data

# check for unique values (to reduce levels)
for c in cc:
    print("Categorical Data = " + c)
    print(churn[c].unique())

# reduce the levels for each category

# replace all occurances of france with "France"
churn.country.replace(["france","Fra"],"France",inplace=True)

# replace all occurances of germany with "Germany"
churn.country.replace(["Ger","germany"],"Germany",inplace=True)

# replace all occurances of spain with "Spain"
churn.country.replace(["Espanio","spain"],"Spain",inplace=True)

# replace all occurances of male to "Male" and female to "Female"
churn.gender.replace(['f','female'],"Female",inplace=True)
churn.gender.replace(['m','M'],"Male",inplace=True)

churn.columns

# distribution of the classes to check imbalance
sns.countplot(x="churn",data=churn)
plt.title("Churn distribution")

# convert the categorical data into dummies
# DV = n-1

# create a copy of the dataset
churn_copy = churn.copy()

# create dummy variables on the categorical data
for c in cc:
    dummy = pd.get_dummies(churn[c],drop_first=True,prefix=c)
    churn_copy = churn_copy.join(dummy)
    
churn_copy.columns

# drop the old categorical data from the dataset
churn_copy.drop(columns=cc,inplace=True)

churn_copy.columns
churn_copy.head(2)

# split data
# split the data
def splitData(data,y,perc=0.3):
    trainx,testx,trainy,testy = train_test_split(data.drop(y,1),
                                                 data[y],
                                                 test_size=perc)
    return(trainx,testx,trainy,testy)

Y = "churn"

# split data into train and test
trainx,testx,trainy,testy = splitData(churn_copy,Y)    

print("trainx={},trainy={},testx={},testy={}".format(trainx.shape,trainy.shape,testx.shape,testy.shape))

trainx.columns

# function to build the logistic regression model using Logit()
def buildModel(trainx,trainy):
    model = sm.Logit(trainy,trainx).fit()
    
    print(model.summary())
    return(model)

# build the model using train data
m1 = buildModel(trainx,trainy)

# cross validation
def doKfoldCV(trainx,trainy,K=5):
    # Accuracy for every CV
    cv_acc = []
    
    # array for CV
    X = trainx.values
    y = trainy.values
    
    # do a 5-fold CV (k=5)
    folds = K
    kf = KFold(folds)
    model = LogisticRegression(solver='liblinear')
    
    # split the data and perform CV
    for train_index, test_index in kf.split(X):
        cv_trainx,cv_trainy = X[train_index], y[train_index]
        cv_testx,cv_testy = X[test_index], y[test_index]
        
        # model and prediction for each combination of train and test
        model.fit(cv_trainx,cv_trainy)
        pred_values = model.predict(cv_testx)
        
        cv_acc.append(accuracy_score(cv_testy,pred_values))
        
    return(cv_acc, sum(cv_acc)/K)

# perform cross-validation
fold_acc,cv_acc = doKfoldCV(trainx,trainy)
fold_acc
print("CV accuracy = {}".format(cv_acc))

# actual (test) predictions
def Predict(model,testx,cutoff=0.5):
    
    # predict on the test data (returns probabilities)
    p1 = model.predict(testx)
    
    # p1 will contain a list of probabilities that need to be converted to classes 0 and 1
    # this is done using the cutoff value
    
    # make a copy of the predictions
    P = p1.copy()
    P[P < cutoff] = 0
    P[P > cutoff] = 1
    
    return(p1,P.astype(int))

# predict on test data
# probs -> probability values of each test row
# predy -> predicted class of each test row
probs,predy = Predict(m1,testx,cutoff=0.25)
predy
probs

# model evaluation
# i) model accuracy
# ii) confusion matrix
# iii) classification report
# iv) AUC / ROC

def cm(actual,predicted):
    
    # 1) model accuracy
    print("Model Accuracy = {}\n".format(accuracy_score(actual,predicted)))
    
    # 2) confusion matrix
    print(pd.crosstab(actual,predicted,margins=True))
    
    # 3) classification report
    print(classification_report(actual,predicted))
    
    # 4) ROC / AUC
    # fpr -> false positive rate
    # tpr -> true positive rate
    
    fpr,tpr,_ = metrics.roc_curve(actual,predicted)
    auc_score = metrics.auc(fpr,tpr)
    
    # plot the ROC curve
    title = "AUC " + str(auc_score)
    plt.plot(fpr,tpr,'b',label=title)
    plt.title("ROC / AUC chart")
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel("TPR")
    plt.xlabel("1-FPR")
    
cm(testy,predy)    
    
    
# class assignment
# do feature selection and select the best features; and build a new model

# feature selection technique (2)
score,pvalue = fs(trainx,trainy)

# create a dataframe to store the features, scores and pval
df_features=pd.DataFrame({'feature':trainx.columns,
                          'score':score,
                          'pval':pvalue})

df_features = df_features.sort_values('score',ascending=False)
df_features    
    
# higher the score, more significant is the feature

    
    
    
    
    
    
    
    



















    












    
    
    
























































