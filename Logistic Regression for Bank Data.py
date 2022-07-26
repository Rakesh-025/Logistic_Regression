import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#Importing Data
df = pd.read_csv(r"C:\Users\kaval\OneDrive\Desktop\Assignments\Assignment Datasets\Logistic Regression\Assignment Datasets\bank_data.csv", sep = ",")
df.columns
df.head(11)
df.describe()
df.info()
df.isna().sum() # no na values

df.columns = 'age', 'default', 'balance', 'housing', 'loan', 'duration', 'campaign','pdays', 'previous', 'poutfailure', 'poutother', 'poutsuccess','poutunknown', 'con_cellular', 'con_telephone', 'con_unknown','divorced', 'married', 'single', 'joadmin', 'joblue_collar','joentrepreneur', 'johousemaid', 'jomanagement', 'joretired','joself_employed', 'joservices', 'jostudent', 'jotechnician','jounemployed', 'jounknown','y' #renaming so that no sapces is there otherwise error.
df = df[['y', 'age', 'default', 'balance', 'housing', 'loan', 'duration', 'campaign','pdays', 'previous', 'poutfailure', 'poutother', 'poutsuccess','poutunknown', 'con_cellular', 'con_telephone', 'con_unknown','divorced', 'married', 'single', 'joadmin', 'joblue_collar','joentrepreneur', 'johousemaid', 'jomanagement', 'joretired','joself_employed', 'joservices', 'jostudent', 'jotechnician','jounemployed', 'jounknown']] # rearranging columns
#############################################################

# Model building 

from sklearn.linear_model import LogisticRegression

X = df.iloc[:,1:]
y = df[["y"]]

log_model = LogisticRegression()
log_model.fit(X, y)

#############################################################

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix

#Model Predictions
y_pred = log_model.predict(X)
y_pred

#Testing Model Accuracy
# Confusion Matrix for the model accuracy
confusion_matrix(y, y_pred)

# The model accuracy is calculated by (a+d)/(a+b+c+d)
accuracy = (39165 + 1177)/(45211) 
accuracy #0.8923049700294177

print(classification_report(y,y_pred)) # accuracy = 0.89

# As accuracy = 0.8923049700294177, which is greater than 0.5; Thus [:,1] Threshold value>0.5=1 else [:,0] Threshold value<0.5=0 
log_model.predict_proba(X)[:,1]

# ROC Curve plotting and finding AUC value
fpr,tpr,thresholds=roc_curve(y,log_model.predict_proba(X)[:,1])
plt.plot(fpr,tpr,color='red')
auc=roc_auc_score(y,y_pred)

plt.plot(fpr,tpr,color='red',label='logit model(area  = %0.2f)'%auc)
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')
plt.show()

print('auc accuracy:',auc) #auc accuracy: 0.6017876828997866 - Average model, it is less than 0.8 

### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
classifier = LogisticRegression(random_state = 0)
model1 = classifier.fit(X_train, y_train)
#Testing model
from sklearn.metrics import confusion_matrix, accuracy_score
y_predtest = classifier.predict(X_test)
print(confusion_matrix(y_test,y_predtest))
print(accuracy_score(y_test,y_predtest)) #Accuracy = 0.8886021822471247 = 88%


#Training model
y_predtrain = classifier.predict(X_train)
print(confusion_matrix(y_train,y_predtrain))
print(accuracy_score(y_train,y_predtrain)) #Accuracy = 0.8933864189338642 = 89%

# train and test accuracy is close enough so it is good model.