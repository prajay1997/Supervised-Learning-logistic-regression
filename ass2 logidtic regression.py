############## solution of Q2) ###############

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

# load the data

data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Logistic  tegression\datasets\advertising.csv")
data.columns
data.info()
# timestamp data has the datatype as string

# lets drop the column which is not useful for analysis

data1 = data.drop(['Ad_Topic_Line','City','Country','Country'], axis =1 )

data1.isnull().sum() # there is no null values

data1.describe()

# convert the timestamp into date time objects
# extract datetime varoables using timestamp column

data1['Timestamp'] = pd.to_datetime(data1['Timestamp']) 

# Converting timestamp column into datatime object in order to extract new features

data1["Month"] = data1["Timestamp"].dt.month
data1["day"] = data1['Timestamp'].dt.day
data1['hour'] = data1["Timestamp"].dt.hour
data1["minutes"] = data1['Timestamp'].dt.minute
data1['weekday'] = data1["Timestamp"].dt.weekday

# drop the timestamp variable to avoid redundancy
data1.drop(['Timestamp'], axis=1, inplace = True)

data1.rename(columns = {'Daily_Time_ Spent _on_Site':'daily_time_spent','Daily Internet Usage':'daily_internet_use'}, inplace = True)
data1.columns

# build the model 
logit_model = sm.logit('Clicked_on_Ad ~ daily_time_spent + Age + Area_Income + daily_internet_use + Male  + Month + day + hour + minutes + weekday', data = data1).fit()
logit_model.summary2()
logit_model.summary()

data1 = data1[['Clicked_on_Ad','daily_time_spent', 'Age', 'Area_Income', 'daily_internet_use', 'Male',
        'Month', 'day', 'hour', 'minutes', 'weekday']]

pred = logit_model.predict(data1.iloc[:, 1:])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(data1.Clicked_on_Ad, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold


import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]


# plot of roc

plt.plot(fpr, tpr)
plt.xlabel("false_positive_rate")
plt.ylabel("True_positive_rate")

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc =auc(fpr,tpr)
print("Area under the ROC curve : %f" % roc_auc)

# filling all the cells with zeroes
data1['pred']= np.zeros(1000)

# taking threshold value and above the prob value will be treated as correct value 

data1.loc[pred > optimal_threshold, "pred"] = 1

classification = classification_report(data1['pred'],data1['Clicked_on_Ad'])
classification

# model building
# split the data into train and test

data_train, data_test = train_test_split(data1, test_size = 0.2, random_state = 100)

model = sm.logit('Clicked_on_Ad ~ daily_time_spent + Age + Area_Income + daily_internet_use + Male  + Month + day + hour + minutes + weekday', data = data_train).fit()
model.summary2()
model.summary()

test_pred = model.predict(data_test)


# Creating new column for storing predicted class of Clicking on Ads
# filling all the cells with zeroes

data_test["test_pred"] = np.zeros(200)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
data_test.loc[test_pred > optimal_threshold, "test_pred"] = 1
data_test["test_pred"].value_counts()

# confusion matrix
confusion_matrix = pd.crosstab(data_test.Clicked_on_Ad, data_test.test_pred)
confusion_matrix

accuracy_test = ( 100+96)/ 200
accuracy_test

classification_test = classification_report(data_test.Clicked_on_Ad, data_test.test_pred)
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(data_test.Clicked_on_Ad, data_test.test_pred)

# plot of Roc
plt.plot(fpr, tpr)
plt.xlabel("false Positive rate")
plt.ylabel("True positive rate")

roc_auc_test = auc(fpr, tpr)
roc_auc_test

# prediction on train data

train_pred = model.predict(data_train)

# creating new column 
# filling it with zeroes

data_train['train_pred'] =np.zeros(800)

# taking threshold value and above the prob value will be treated as correct value 

data_train.loc[train_pred > optimal_threshold, "train_pred"] = 1
data_train["train_pred"].value_counts()

# confusion matrix
confusion_matrix_train = pd.crosstab(data_train.Clicked_on_Ad, data_train.train_pred)
confusion_matrix_train

accuracy_test = (391+385)/800
accuracy_test
