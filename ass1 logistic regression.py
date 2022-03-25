### solution of Q1) ###############


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

# import data 
data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Logistic  tegression\datasets\Affairs.csv")

# convert the naffairs column into binary form 

data.loc[data.naffairs >= 1, "naffairs"] = 1

data['naffairs'].value_counts()
data.columns
# removing unnamed column
data.drop(['Unnamed: 0'],axis=1 , inplace = True)

data.info()
data.describe()
 
# graphical representation
import seaborn as sns
sns.countplot(data.naffairs)
sns.countplot(data.kids)
sns.countplot(x ="naffairs", hue = "kids", data = data)

# very few people having the affair and those who have not kids they only have naffairs

data.corr()
plt.figure(figsize=(20,12))
sns.heatmap(data.corr().round(2), annot = True)
data.columns
# model_building 
logit_model = sm.logit("naffairs ~ kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5 + yrsmarr6", data = data).fit() 

logit_model.summary2()    
logit_model.summary()      

pred = logit_model.predict(data.iloc[:,1:])

fpr, tpr, thresholds = roc_curve(data.naffairs, pred)
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
data['pred']= np.zeros(601)

# taking threshold value and above the prob value will be treated as correct value 

data.loc[pred > optimal_threshold, "pred"] = 1

# classification report
classification = classification_report(data["pred"], data["naffairs"])
classification


data["pred"].value_counts()

# model building
### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(c1, test_size = 0.2) # 30% test data
model = sm.logit("naffairs ~ kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5 + yrsmarr6", data = train_data).fit() 

model.summary2()
model.summary()

test_pred = model.predict(test_data)

# Creating new column for storing predicted class of naffairs
# filling all the cells with zeroes

test_data["test_pred"] = np.zeros(121)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1
test_data["test_pred"].value_counts()

# confusion matrix
confusion_matrix = pd.crosstab(test_data.naffairs, test_data.test_pred)
confusion_matrix

accuracy_test = ( 70 + 23)/ 121
accuracy_test

classification_test = classification_report(test_data.test_pred, test_data.naffairs)
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data.naffairs, test_pred)

# plot of Roc
plt.plot(fpr, tpr)
plt.xlabel("false Positive rate")
plt.ylabel("True positive rate")

roc_auc_test = auc(fpr, tpr)
roc_auc_test

# prediction on train data

train_pred = model.predict(train_data)

# creating new column 
# filling it with zeroes

train_data['pred'] =np.zeros(480)

# taking threshold value and above the prob value will be treated as correct value 

train_data.loc[train_pred > optimal_threshold, "pred"] = 1
train_data["pred"].value_counts()

# confusion matrix
confusion_matrix_train = pd.crosstab(train_data.naffairs, train_data.pred)
confusion_matrix_train

accuracy_test = ( 243 + 77)/480
accuracy_test
