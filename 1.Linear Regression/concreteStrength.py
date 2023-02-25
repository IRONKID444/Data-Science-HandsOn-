# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 09:13:37 2022

@author: Yash Pungaliya
"""

#Singualrity :- the column which have 85% or more same value do not  predictive values
#Steps in EDA : 1)Correlation 2)Outliers using Boxplot 3)Normalization check using histogram 4)Checking and fixing null values
#Linear Regression
#Dataset : - Concrete 

#import libraries
import pandas as pd 
import numpy as np 
import statsmodels.api as sm  #OLS model
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats  as stats
from sklearn.model_selection import train_test_split
import statsmodels.stats.api as sms #heteroscedasticity check
import pylab#plot to check hetero or homo scedasticiity 
from sklearn import preprocessing # data transformation
from sklearn.metrics import mean_squared_error

#read data
filename = "C:/Users/Yash Pungaliya/Desktop/Linear Regression/concrete.csv"
conc = pd.read_csv(filename)
print (conc.head)

conc.shape
conc.columns
conc.rename(columns={'cementcomp':'ccomp',
                     'superplastisizer':'super',
                     'coraseaggr': 'caggr',
                     'finraggr':'faggr'})

conc.isnull().sum()
conc[conc==0].count()

#summary of the data
conc.describe()

#check for outliers in age
conc.age.describe()

#extract 'Y' variable from 
Y = "CCS"
features = list(conc.columns)
features.remove(Y)
features


#function to check normality 
#input : numeric data
#O/p : returns statsu of each cilumn as Normal or not Normal 
#test: Shapiro/Agustion Test

def checkNormality(data,col):
    shapiro = []
    agistino = []
    
    for c in col:
        tstat,pval = stats.shapiro(data[c])
        if pval < 0.05:
            shapiro.append('Not Normal')
        else:
            shapiro.append('Normal')

    for c in col:
        tstat,pval = stats.normaltest(data[c])
        if pval < 0.05:
            agistino.append('Not Normal')
        else:
            agistino.append('Normal')
            
   
    nd=pd.DataFrame({'features':col,
                             'shapiro':shapiro,
                             'agistino':agistino})
    return(nd)

checkNormality(conc, features)
# =============================================================================
# conc.shape
# features
# =============================================================================

#b:boxplot h:histogram  hm = heatmap 
def plotCharts(data,nc,ctype):
    if ctype in ['b','h','hm']:
        
        #histogram /boxplot
        if ctype in ['h','b']:
            ROWS = int(np.ceil(len(nc)/2))
            COLS = 2
            POS = 1
            
            #set the font scale and outer figures
            sns.set(font_scale =1 ,color_codes = True)
            fig = plt.figure()
            
            for c in nc:
                fig.add_subplot(ROWS,COLS,POS)
                if(ctype=="h"):
                    sns.distplot(data[c]).set_ylabel(c)
                else:
                    sns.boxplot(data[c]).set_ylabel(c)
                POS+=1
        else:
            #heatmap
            cor = data[nc].corr()
            cor = np.tril(cor) #Fill the upper triangle matrix with 0
            print(sns.heatmap(cor, xticklabels = nc, yticklabels=nc,
                        annot=True,vmin =-1,vmax=1))
    else:
        print('Invalid Chart type'+ctype)
        
# Plot Individual Charts         
plotCharts(conc, features, "h")
plotCharts(conc, features, "b")
plotCharts(conc, features, "hm")



#Slipt the Data
def splitData(data,Y,perc=0.3):
    trainx,testx,trainY,testY = train_test_split(data.drop(Y,1),data[Y],test_size=0.3)
    
    return(trainx,testx,trainY,testY )

trainx,testx,trainY,testY  = splitData(conc, Y)
print("trainx={} , trainY = {}".format(trainx.shape,trainY.shape))
print("testX={} , testY = {}".format(testx.shape,testY.shape))

#Build Model 
def buildModel(trainx,trainy):
    model = sm.OLS(trainY,trainx).fit()
    print(model.summary())
    return(model)

#Model 1 
m1 = buildModel(trainx, trainY)     

#validate the assumptions of linear regression
#1)mean of residuals is 0
np.mean(m1.resid)   


def checkHeteroscedasticity(model,trainY):
    
    sns.set(style="whitegrid")
    sns.residplot(x=m1.resid,y=trainY,lowess= True ,color = 'blue')
    
    #Breusch Pagan Test 
    pvalue = sms.het_breuschpagan(m1.resid,m1.model.exog)[1]
    if pvalue <0.05:
        return("Model is Heteroscedastic")
    else:
        return("Model is Homoscedastic")


checkHeteroscedasticity(m1, trainY)

#Prediction 
def prediction(model,testx,testY):
    pred = round(model.predict(testx),2)
    
    
    #Store Actual and predicted data for  analysis
    df = pd.DataFrame({'actual':testY ,'predicted':pred})
    df['err']=df.actual-df.predicted
    
    #MSE 
    mse=mean_squared_error(testY, pred)
    return(df,mse)
res1,mse1 = prediction(m1,testx,testY)

print("Model 1 mse ={}" .format(mse1))

print(mse1)
