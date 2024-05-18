#!/usr/bin/env python
# coding: utf-8

# In[309]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import pickle
import csv


# In[290]:


#Function to calculate Z score
def calculateZScore(row):
    card_number = row['Card Number']
    mean_amount = cardStats[card_number]['Mean Amount']
    std_amount = cardStats[card_number]['Std Amount']
    zScore = (row['Amount'] - mean_amount) / std_amount
    return zScore


# In[291]:


#Read from file, change columns
file_path1 = r"C:\Users\youss\Desktop\FraudDetection\fraudTrain.csv"
file_path2 = r"C:\Users\youss\Desktop\FraudDetection\fraudTest.csv"

df = pd.read_csv(file_path1)
#df2 = pd.read_csv(file_path2)
#df = pd.concat([df1, df2], ignore_index=True)

df.set_index('trans_num', inplace = True)
df = df.drop('ID', axis = 1)
df = df.drop('firstName', axis = 1)
df = df.drop('lastName', axis = 1)
df = df.drop('merchant', axis = 1) #Remove merchant, too many different labels


# In[292]:


#Change time string to object
df["Time"] = pd.to_datetime(df["Time"], errors='coerce')
df.dropna(inplace=True)


# In[293]:


#Split time to Day of week, hour, and month
df['Hour of Day'] = df['Time'].dt.hour
df['Day of Week'] = df['Time'].dt.dayofweek
df['Month'] = df['Time'].dt.month

#Turn category into labels
label_encoder = LabelEncoder()
df['Category'] = label_encoder.fit_transform(df['category'])
for x in range(len(list(label_encoder.classes_))):
    print(x, list(label_encoder.classes_)[x])


# In[294]:


#Group data by card num
grouped_by_card = df.groupby('Card Number')
cardStats = {}

#calculate standard deviation for each card
for card, data in grouped_by_card:
    mean = data['Amount'].mean()
    std = data['Amount'].std()
    cardStats[card] = {'Mean Amount': mean, 'Std Amount': std}

#save data into a dictionary to potentially be used
cardStatsDf = pd.DataFrame.from_dict(cardStats, orient='index')
cardStatsDf.to_csv('CardStats.csv')


# In[295]:


#Apply Z Score Function
df['Z Score'] = df.apply(calculateZScore, axis = 1)


# In[296]:


#sort data by time
df_sorted = df.sort_values(by=['Card Number', 'Time'])

#.diff() calculates the difference between consecutive entries
df['Time Difference'] = df_sorted.groupby('Card Number')['Time'].diff().dt.total_seconds()
median = df['Time Difference'].median()
df['Time Difference'] = df['Time Difference'].fillna(median)

#Log transform and normalize, try different combinations to find best accuracy
scaler = MinMaxScaler()
df['Log Time Difference'] = np.log(df['Time Difference'] + 1) #adding 1 to deal with log(0)
df['Normalized Log Time Difference'] = scaler.fit_transform(df[['Time Difference']])


# In[297]:


#Log transform and normalize, try different combinations to find best accuracy
scaler = MinMaxScaler()
df['Log Transformed Amount'] = np.log10(df['Amount'])
df['Normalized Log Amount'] = scaler.fit_transform(df[['Log Transformed Amount']])


# In[298]:


#drop unneeded data
df = df.drop("Amount", axis = 1)
df = df.drop("Time", axis = 1)
df = df.drop("Card Number", axis = 1)
df = df.drop("Log Transformed Amount", axis = 1)
df = df.drop("Time Difference", axis = 1)
df = df.drop("Log Time Difference", axis = 1)
df = df.drop("category", axis = 1)


# In[299]:


X = df[['Category', 'Hour of Day', 'Day of Week', 'Month', 'Z Score', 'Normalized Log Time Difference', 'Normalized Log Amount']]
y = df['is_fraud']


# In[300]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


# In[301]:


xgb_model = xgb.XGBClassifier(scale_pos_weight=(len(y_train)-y_train.sum())/y_train.sum(), alpha = 2.1, learning_rate = 0.4)
xgb_model.fit(X_train, y_train)


# In[302]:


custom_threshold = 0.1
y_val_proba = xgb_model.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_proba > custom_threshold).astype(int)


# In[303]:


conf_matrix = confusion_matrix(y_val, y_val_pred)

TP = conf_matrix[1, 1]
FP = conf_matrix[0, 1]
TN = conf_matrix[0, 0]
FN = conf_matrix[1, 0]

accuracy = (TP + TN) / (TP + FP + TN + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1_score = (precision * recall) / (precision + recall)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {F1_score}")
print(f"TP: {TP}")
print(f"TN: {TN}")
print(f"FP: {FP}")
print(f"FN: {FN}")


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

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {F1_score}")
print(f"TP: {TP}")
print(f"TN: {TN}")
print(f"FP: {FP}")
print(f"FN: {FN}")

data = [TP, FP, TN, FN]
with open('metrics.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(data)


# In[306]:


filename='trained_model.sav'
pickle.dump(xgb_model,open('trained_model.sav','wb'))
df.to_pickle('data.pkl')


# threshhold = 0.01 alpha = 5 learning rate = 0.3
# Accuracy: 0.9370475149809651
# Precision: 0.08219277810133954
# Recall: 0.9912203687445127
# F1 Score: 0.07589915966386554
# TP: 1129
# TN: 186675
# FP: 12607
# FN: 10
# 
# threshhold = 0.1 alpha = 5 learning rate = 0.3
# Accuracy: 0.9767090274971185
# Precision: 0.19125109361329834
# Recall: 0.9596136962247586
# F1 Score: 0.15946892325649256
# TP: 1093
# TN: 194660
# FP: 4622
# FN: 46
# 
# threshhold = 0.3 alpha = 5 learning rate = 0.3
# Accuracy: 0.9868526751188748
# Precision: 0.29427942794279427
# Recall: 0.9394205443371378
# F1 Score: 0.22408376963350785
# TP: 1070
# TN: 196716
# FP: 2566
# FN: 69
# 
# threshhold = 0.5 alpha = 5 learning rate = 0.3
# Accuracy: 0.9903952180659711
# Precision: 0.36401384083044985
# Recall: 0.9236172080772608
# F1 Score: 0.2611069744353438
# TP: 1052
# TN: 197444
# FP: 1838
# FN: 87
# 
# threshhold = 0.99 alpha = 5 learning rate = 0.3
# Accuracy: 0.9976100308849871
# Precision: 0.8473684210526315
# Recall: 0.7067603160667252
# F1 Score: 0.3853518429870752
# TP: 805
# TN: 199137
# FP: 145
# FN: 334
# 
# Higher threshhold increases precision by decreasing FP but increases FN. Best option depends on the "cost" of each.

# threshhold = 0.01 alpha = 5 learning rate = 0.1
# Accuracy: 0.9373967797785662
# Precision: 0.08273591806876372
# Recall: 0.9929762949956101
# F1 Score: 0.07637247619690729
# TP: 1131
# TN: 186743
# FP: 12539
# FN: 8
# 
# threshhold = 0.01 alpha = 5 learning rate = 0.2
# Accuracy: 0.9646893289625339
# Precision: 0.1366862457170827
# Recall: 0.9806848112379281
# F1 Score: 0.11996563204811513
# TP: 1117
# TN: 192227
# FP: 7055
# FN: 22
# 
# threshhold = 0.01 alpha = 5 learning rate = 0.3
# Accuracy: 0.9767090274971185
# Precision: 0.19125109361329834
# Recall: 0.9596136962247586
# F1 Score: 0.15946892325649256
# TP: 1093
# TN: 194660
# FP: 4622
# FN: 46
# 
# threshhold = 0.01 alpha = 5 learning rate = 0.4
# Accuracy: 0.9823870752066899
# Precision: 0.2374258730507358
# Recall: 0.9490781387181738
# F1 Score: 0.18991567111735772
# TP: 1081
# TN: 195810
# FP: 3472
# FN: 58
# 
# threshhold = 0.01 alpha = 5 learning rate = 0.5
# Accuracy: 0.9863637044022333
# Precision: 0.28609769189479334
# Recall: 0.935908691834943
# F1 Score: 0.21911613566289823
# TP: 1066
# TN: 196622
# FP: 2660
# FN: 73
# 
# Similiar to threshhold. Higher learning rate means better precision **and** accuracy. FP decreaase but FN increase. Best option depends on the "cost" of each.

# threshhold = 0.1 alpha = 1 learning rate = 0.4
# Accuracy: 0.9825467391141647
# Precision: 0.23841206475937016
# Recall: 0.9438103599648815
# F1 Score: 0.1903328611898017
# TP: 1075
# TN: 195848
# FP: 3434
# FN: 64
# 
# threshhold = 0.1 alpha = 2 learning rate = 0.4
# Accuracy: 0.9829458988828516
# Precision: 0.24341364557532089
# Recall: 0.9490781387181738
# F1 Score: 0.19372759856630825
# TP: 1081
# TN: 195922
# FP: 3360
# FN: 58
# 
# threshhold = 0.1 alpha = 3 learning rate = 0.4
# Accuracy: 0.9821475793454778
# Precision: 0.23483365949119372
# Recall: 0.9482001755926251
# F1 Score: 0.18821889159986055
# TP: 1080
# TN: 195763
# FP: 3519
# FN: 59
# 
# threshhold = 0.1 alpha = 1.9 learning rate = 0.4
# Accuracy: 0.9826265710679021
# Precision: 0.23926107277987982
# Recall: 0.9438103599648815
# F1 Score: 0.19087357954545456
# TP: 1075
# TN: 195864
# FP: 3418
# FN: 64
# 
# threshhold = 0.1 alpha = 2.1 learning rate = 0.4
# Accuracy: 0.9825617076054904
# Precision: 0.2398409893992933
# Recall: 0.9534679543459175
# F1 Score: 0.19163578613022764
# TP: 1086
# TN: 195840
# FP: 3442
# FN: 53

# In[307]:


df.head()

