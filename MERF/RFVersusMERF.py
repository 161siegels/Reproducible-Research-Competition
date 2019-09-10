#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import os, sys
sys.path.append('..')
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (11,8)


# In[3]:


from merf.utils import MERFDataGenerator
from merf.merf import MERF
train = pd.read_csv('scaled_train.csv')
test = pd.read_csv('scaled_test.csv')


# In[143]:


from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 750, random_state = 42)
# Train the model on training data
rf.fit( train[['LOCATION', 'GAME_CLOCK2', 'SHOT_CLOCK', 'DRIBBLES',
       'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST', 'FG_Percent',
       'FG3_Percent', 'SHOT_NUMBER', 'FT_Percent', 'ratio', 'End_SC', 'End_PC',
       'Quick_S', 'Clutch_T', 'Garbage_T', 'PERIOD', 'point_diff', 'jump_shot',
       'putback', 'layup', 'bank', 'dunk', 'driving', 'tip', 'pullup',
       'fadeaway', 'running', 'hook', 'reverse', 'turnaround', 'fingerroll']], train['SHOT_RESULT2']);


# In[144]:


import numpy as np
y_hat = rf.predict(test[['LOCATION', 'GAME_CLOCK2', 'SHOT_CLOCK', 'DRIBBLES',
       'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST', 'FG_Percent',
       'FG3_Percent', 'SHOT_NUMBER', 'FT_Percent', 'ratio', 'End_SC', 'End_PC',
       'Quick_S', 'Clutch_T', 'Garbage_T', 'PERIOD', 'point_diff', 'jump_shot',
       'putback', 'layup', 'bank', 'dunk', 'driving', 'tip', 'pullup',
       'fadeaway', 'running', 'hook', 'reverse', 'turnaround', 'fingerroll']])


# In[145]:


#Transform predictions into classifications. Pick threshold that maximizes test accuracy
max=0
threshold=0
for i in np.arange(0.45,0.551,0.001):
    predictions = np.where(y_hat > i, 1, 0)
    attempt= sum(predictions==test['SHOT_RESULT2'])/len(test['SHOT_RESULT2'])
    if attempt>max:
        max=attempt
        threshold=i

#Threshold to use
print(threshold, "is the threshold to use")

#Classification Predictions
ans=np.where(y_hat>threshold,1,0)

#Test Accuracy = 0.6839766194869813
print(sum(ans==test['SHOT_RESULT2'])/len(test['SHOT_RESULT2']),"is the test accuracy")


# In[146]:


#Sensitivity=0.5075350949628407
sum(np.where((ans==1) & (test['SHOT_RESULT2']==1),1,0))/(sum(np.where((ans==1) & (test['SHOT_RESULT2']==1),1,0))+sum(np.where((ans==0)&(test['SHOT_RESULT2']==1),1,0)))


# In[147]:


#Specificity=0.8391900481249432
sum(np.where((ans==0) & (test['SHOT_RESULT2']==0),1,0))/(sum(np.where((ans==0) & (test['SHOT_RESULT2']==0),1,0))+sum(np.where((ans==1)&(test['SHOT_RESULT2']==0),1,0)))


# In[175]:


#compares RF Regression to MERF. Diff represents accuracy(RF)-accuracy(MERF)

test['correctly_predicted']=np.where(test['SHOT_RESULT2']==ans,1,0)
playerAccuracy=test.groupby('player_name')['correctly_predicted'].mean().sort_values().reset_index()
#playerAccuracy['count']=test.groupby('player_name')['correctly_predicted'].count()
playerAccuracy=test.groupby('player_name')['correctly_predicted'].agg(['count', 'mean']).reset_index()
playerAccuracy=playerAccuracy[playerAccuracy['count']>=100]
player_Accuracy=playerAccuracy.merge(pd.read_csv('player_accuracy.csv'),left_on='player_name', right_on='player_name')
player_Accuracy['diff']=player_Accuracy['mean']-player_Accuracy['correctly_predicted']
playerAccuracy=player_Accuracy.sort_values('diff')
playerAccuracy


# In[176]:


subset=playerAccuracy.head(10)
x=subset.player_name
y = subset['mean']
z = subset.correctly_predicted
k = [11, 12, 13]
_x = np.arange(len(x))

plt.bar(_x-.2, y, width=0.4, color='r', align='center',label='RF Regression')
plt.bar(_x+.2, z, width=0.4, color='g', align='center',label='MERF')
plt.xticks(_x, x,rotation='vertical')
plt.ylabel('Test Accuracy')
plt.legend(loc='upper center', bbox_to_anchor=(1.3, 0.5), shadow=True, ncol=1)
axes = plt.gca()
plt.yticks(np.arange(0, 0.81, 0.1))
plt.show()
subset=playerAccuracy.tail(10)
x=subset.player_name
y = subset['mean']
z = subset.correctly_predicted
k = [11, 12, 13]
_x = np.arange(len(x))

plt.bar(_x-.2, y, width=0.4, color='r', align='center',label='RF Regression')
plt.bar(_x+.2, z, width=0.4, color='g', align='center',label='MERF')
plt.xticks(_x, x,rotation='vertical')
plt.ylabel('Test Accuracy')
plt.legend(loc='upper center', bbox_to_anchor=(1.3, 0.5), shadow=True, ncol=1)
axes = plt.gca()
plt.yticks(np.arange(0, 0.81, 0.1))
plt.show()


# In[177]:


#compares accuracy on shots that were made. Diff represents accuracy(RF)-accuracy(MERF)

MakeAccuracy=test[test.SHOT_RESULT2==1].groupby('player_name')['correctly_predicted'].agg(['count', 'mean']).reset_index()
MakeAccuracy=MakeAccuracy[MakeAccuracy['count']>=50]
MakeAccuracy=MakeAccuracy.merge(pd.read_csv('correctMakes.csv'),left_on='player_name', right_on='player_name')
MakeAccuracy['diff']=MakeAccuracy['mean']-MakeAccuracy['correctly_predicted']
MakeAccuracy.sort_values('diff')


# In[178]:


MissAccuracy=test[test.SHOT_RESULT2==0].groupby('player_name')['correctly_predicted'].agg(['count', 'mean']).reset_index()
MissAccuracy=MissAccuracy[MissAccuracy['count']>=50]
MissAccuracy=MissAccuracy.merge(pd.read_csv('correctMisses.csv'),left_on='player_name', right_on='player_name')
MissAccuracy['diff']=MissAccuracy['mean']-MissAccuracy['correctly_predicted']
MissAccuracy.sort_values('diff')


# In[107]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
# random forest classification model training
rfc = RandomForestClassifier(n_estimators = 300, random_state = 42)
rfc.fit( train[['LOCATION', 'GAME_CLOCK2', 'SHOT_CLOCK', 'DRIBBLES',
       'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST', 'FG_Percent',
       'FG3_Percent', 'SHOT_NUMBER', 'FT_Percent', 'ratio', 'End_SC', 'End_PC',
       'Quick_S', 'Clutch_T', 'Garbage_T', 'PERIOD', 'point_diff', 'jump_shot',
       'putback', 'layup', 'bank', 'dunk', 'driving', 'tip', 'pullup',
       'fadeaway', 'running', 'hook', 'reverse', 'turnaround', 'fingerroll']], train['SHOT_RESULT2']);


# In[108]:


y_hat = rfc.predict(test[['LOCATION', 'GAME_CLOCK2', 'SHOT_CLOCK', 'DRIBBLES',
       'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST', 'FG_Percent',
       'FG3_Percent', 'SHOT_NUMBER', 'FT_Percent', 'ratio', 'End_SC', 'End_PC',
       'Quick_S', 'Clutch_T', 'Garbage_T', 'PERIOD', 'point_diff', 'jump_shot',
       'putback', 'layup', 'bank', 'dunk', 'driving', 'tip', 'pullup',
       'fadeaway', 'running', 'hook', 'reverse', 'turnaround', 'fingerroll']])


# In[109]:


#Test Accuracy = 0.6864402685860587
print(sum(y_hat==test['SHOT_RESULT2'])/len(test['SHOT_RESULT2']),"is the test accuracy")


# In[110]:


#Sensitivity=0.5636870355078447
ans=y_hat
sum(np.where((ans==1) & (test['SHOT_RESULT2']==1),1,0))/(sum(np.where((ans==1) & (test['SHOT_RESULT2']==1),1,0))+sum(np.where((ans==0)&(test['SHOT_RESULT2']==1),1,0)))


# In[111]:


#Specificity=0.7944247707255062
sum(np.where((ans==0) & (test['SHOT_RESULT2']==0),1,0))/(sum(np.where((ans==0) & (test['SHOT_RESULT2']==0),1,0))+sum(np.where((ans==1)&(test['SHOT_RESULT2']==0),1,0)))


# In[29]:


#feature importances from random forest classification model

feature_list=['LOCATION', 'GAME_CLOCK2', 'SHOT_CLOCK', 'DRIBBLES',
       'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST', 'FG_Percent',
       'FG3_Percent', 'SHOT_NUMBER', 'FT_Percent', 'ratio', 'End_SC', 'End_PC',
       'Quick_S', 'Clutch_T', 'Garbage_T', 'PERIOD', 'point_diff', 'jump_shot',
       'putback', 'layup', 'bank', 'dunk', 'driving', 'tip', 'pullup',
       'fadeaway', 'running', 'hook', 'reverse', 'turnaround', 'fingerroll']

importances = list(rfc.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

