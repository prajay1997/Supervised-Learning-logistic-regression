############## solution of Q3) ###############

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as sm
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

# load the datasets

vote =pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Logistic  tegression\datasets\election_data.csv")
vote.columns
# dropping the colums wichi is not useful
vote.drop(['Election-id'], axis =1, inplace= True)
vote1.info()
vote1 = vote.rename(columns = {'Amount Spent':'amount_spent','Popularity Rank':'popularity_rank'})
vote1.describe()

vote1.isnull().sum()
# there is null value in each column
vote1.fillna(vote1.median(), inplace = True) 
vote1.isnull().sum()

# built the model
vote1.columns
logit_model = sm.logit('Result ~  amount_spent +  popularity_rank ', data = vote1).fit()

logit_model.summary2()
logit_model.summary()

pred = logit_model.predict(vote1.iloc[:,1:])

fpr,tpr, thresholds = roc_curve(vote1.Result, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# plot roc curve 
plt.plot(fpr, tpr)
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")


# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print(" area under the curve:%f" %roc_auc)

# filling all the cells with zeroes
vote1['pred'] =  np.zeros(11)
# taking threshold value and above the prob value is the correct value

vote1.loc[pred > optimal_threshold, "pred"] = 1

classification = classification_report(vote1["Result"], vote1["pred"])
classification

# splitting the data into train and test data

train_vote, test_vote =  train_test_split(vote1, test_size = 0.2)

model = sm.logit('Result ~  amount_spent +  popularity_rank ', data = train_vote).fit()
model.summary2()
model.summary()

test_pred = model.predict(test_vote.iloc[:,1:])

# craeting new column for sorting the predicted class of Result
# filling all the cells with zeroes
test_vote['test_pred'] = np.zeros(3)
# taking threshold value as optimal_threshold and above threshold prob value wil be treated as 1
test_vote.loc[test_pred > optimal_threshold,"test_pred"] = 1 

# confusion metrics
 pd.crosstab(test_vote.Result, test_vote.test_pred)

accuracy_test = (1+1)/3
accuracy_test

Classification_test = classification_report(test_vote.Result, test_vote.test_pred)
Classification_test 

##ROC CURVE AND AUC
fpr, tpr, threshold =  metrics.roc_curve(test_vote.Result, test_pred)

#PLOT OF ROC
pl.plot(fpr, tpr)
pl.xlabel("False positive rate")
pl.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test

# prediction on train datasets

train_pred = model.predict(train_vote.iloc[:, 1:])

# creating new column for sorting the predicted class of Result
# filling all the cells with zeroes

train_vote["train_pred"] = np.zeros(8)
# taking threshold value and above the prob value will be treated as correct value 

train_vote.loc[train_pred > optimal_threshold, 'train_pred'] =1

pd.crosstab(train_vote.Result, train_vote.train_pred)
accuracy_train = (3+4)/8
accuracy_train
