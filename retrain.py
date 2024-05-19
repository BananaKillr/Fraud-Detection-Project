import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import confusion_matrix
import pickle
import csv

# In[299]:
df = pd.read_pickle('data.pkl')

X = df[['Category', 'Hour of Day', 'Month', 'Z Score', 'Normalized Log Time Difference', 'Normalized Log Amount']]
y = df['is_fraud']


# In[300]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


# In[301]:


xgb_model = xgb.XGBClassifier(scale_pos_weight=(len(y_train)-y_train.sum())/y_train.sum(), alpha = 0.1, learning_rate = 0.1)
xgb_model.fit(X_train, y_train)


# In[302]:


custom_threshold = 0.3
y_val_proba = xgb_model.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_proba > custom_threshold).astype(int)


# In[304]:


y_test_proba = xgb_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba > custom_threshold).astype(int)


# In[310]:
conf_matrix = confusion_matrix(y_test, y_test_pred)

TP = conf_matrix[1, 1]
FP = conf_matrix[0, 1]
TN = conf_matrix[0, 0]
FN = conf_matrix[1, 0]

accuracy = (TP + TN) / (TP + FP + TN + FN) #(0+2)/(0+1+2+3)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1_score = (precision * recall) / (precision + recall)

data = [TP, FP, TN, FN]
with open('metrics.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(data)


# In[306]:


filename='trained_model.sav'
pickle.dump(xgb_model,open('trained_model.sav','wb'))
df.to_pickle('data.pkl')