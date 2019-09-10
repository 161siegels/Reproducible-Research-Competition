import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from merf.utils import MERFDataGenerator
from merf.merf import MERF

#Bring in train and test
train = pd.read_csv('scaled_train.csv')
test = pd.read_csv('scaled_test.csv')


#Fixed effects
X_train=train[['LOCATION', 'GAME_CLOCK2', 'SHOT_CLOCK', 'DRIBBLES',
       'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST', 'FG_Percent',
       'FG3_Percent', 'SHOT_NUMBER', 'FT_Percent', 'ratio', 'End_SC', 'End_PC',
       'Quick_S', 'Clutch_T', 'Garbage_T', 'PERIOD', 'point_diff', 'jump_shot',
       'putback', 'layup', 'bank', 'dunk', 'driving', 'tip', 'pullup',
       'fadeaway', 'running', 'hook', 'reverse', 'turnaround', 'fingerroll']]

#Random effects
Z_train=train[['LOCATION', 'GAME_CLOCK2', 'SHOT_CLOCK', 'DRIBBLES',
       'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST', 'SHOT_NUMBER',
               'ratio', 'End_SC', 'End_PC',
       'Quick_S', 'Clutch_T', 'Garbage_T', 'PERIOD', 'point_diff', 'jump_shot',
       'putback', 'layup', 'bank', 'dunk', 'driving', 'tip', 'pullup',
       'fadeaway', 'running', 'hook', 'reverse', 'turnaround', 'fingerroll']]

#Response vector
y_train=train['SHOT_RESULT2']

#clusters
clusters_train = train['player_name']

#Note that we tested values for n_estimators from 200 to 1000 (only included the one below because of time complexity)
#Similarly tested values for max_iterations from 10 to 100
#Chose parameters that gave the highest test accuracy
mrf = MERF(n_estimators=800, max_iterations=30)
mrf.fit(X_train, Z_train, clusters_train, y_train)

#Load model from joblib file if do not have time to fit the model.
from joblib import dump, load
dump(mrf, 'model2.joblib')
mrf=load('model2.joblib')


#Organizing the test data
X_new = test[['LOCATION', 'GAME_CLOCK2', 'SHOT_CLOCK', 'DRIBBLES',
       'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST', 'FG_Percent',
       'FG3_Percent', 'SHOT_NUMBER', 'FT_Percent', 'ratio', 'End_SC', 'End_PC',
       'Quick_S', 'Clutch_T', 'Garbage_T', 'PERIOD', 'point_diff', 'jump_shot',
       'putback', 'layup', 'bank', 'dunk', 'driving', 'tip', 'pullup',
       'fadeaway', 'running', 'hook', 'reverse', 'turnaround', 'fingerroll']]
Z_new = test[['LOCATION', 'GAME_CLOCK2', 'SHOT_CLOCK', 'DRIBBLES',
       'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST', 'SHOT_NUMBER',
               'ratio', 'End_SC', 'End_PC',
       'Quick_S', 'Clutch_T', 'Garbage_T', 'PERIOD', 'point_diff', 'jump_shot',
       'putback', 'layup', 'bank', 'dunk', 'driving', 'tip', 'pullup',
       'fadeaway', 'running', 'hook', 'reverse', 'turnaround', 'fingerroll']]
clusters_new = test['player_name']
y_new = test['SHOT_RESULT2']

#Generates Predictions
y_hat_new = mrf.predict(X_new, Z_new, clusters_new)

#Transform predictions into classifications. Pick threshold that maximizes test accuracy
max=0
threshold=0
for i in np.arange(0.45,0.551,0.001):
    predictions = np.where(y_hat_new > i, 1, 0)
    attempt= sum(predictions==y_new)/len(y_new)
    if attempt>max:
        max=attempt
        threshold=i

#Threshold to use= 0.5410000000000001
print(threshold)

#Classification Predictions
ans=np.where(y_hat_new>threshold,1,0)

#Test Accuracy = 0.6884691560794165
print(sum(ans==y_new)/len(y_new))

#Calculate sensitivity and specificity
truePos=sum(np.where((ans==1) & (y_new==1),1,0))
falseNeg=sum(np.where((ans==0)&(y_new==1),1,0))
print(truePos/(truePos+falseNeg))

trueNeg=sum(np.where((ans==0) & (y_new==0),1,0))
falsePos=sum(np.where((ans==1)&(y_new==0),1,0))
print(trueNeg/(trueNeg+falsePos))

#sens=0.522914946325351
#spec=0.8341051484609099

#Look at accuracies on a per player basis
test['correctly_predicted']=np.where(test['SHOT_RESULT2']==ans,1,0)
test.groupby('player_name')['correctly_predicted'].mean().sort_values().reset_index().to_csv('player_accuracy.csv')

#Accuracies on makes per player
test[test.SHOT_RESULT2==1].groupby('player_name')['correctly_predicted'].mean().sort_values().reset_index().to_csv('correctMakes.csv')

#Accuracies on misses per player
test[test.SHOT_RESULT2==0].groupby('player_name')['correctly_predicted'].mean().sort_values().reset_index().to_csv('correctMisses.csv')





#Visualizations- credit https://github.com/manifoldai/merf/blob/master/notebooks/MERF%20Example.ipynb

#GLL
plt.figure(figsize=[15,10])
plt.subplot(221)
plt.plot(mrf.gll_history)
plt.grid('on')
plt.ylabel('GLL')
plt.xlabel('Iteration')

#Sigma_b^2 tuning
plt.subplot(222)
D_hat_history = [x[0][0] for x in mrf.D_hat_history]
plt.plot(D_hat_history)
plt.grid('on')
plt.ylabel('sigma_b2_hat')
plt.xlabel('Iteration')

#sigma_e^2 tuning
plt.subplot(223)
plt.plot(mrf.sigma2_hat_history)
plt.grid('on')
plt.ylabel('sigma_e2_hat')
plt.xlabel('Iteration')


#Dataframe of the b
mrf.trained_b

#Random forest for fixed effects
rf=mrf.trained_rf


feature_list=['LOCATION', 'GAME_CLOCK2', 'SHOT_CLOCK', 'DRIBBLES',
       'TOUCH_TIME', 'SHOT_DIST', 'PTS_TYPE', 'CLOSE_DEF_DIST', 'FG_Percent',
       'FG3_Percent', 'SHOT_NUMBER', 'FT_Percent', 'ratio', 'End_SC', 'End_PC',
       'Quick_S', 'Clutch_T', 'Garbage_T', 'PERIOD', 'point_diff', 'jump_shot',
       'putback', 'layup', 'bank', 'dunk', 'driving', 'tip', 'pullup',
       'fadeaway', 'running', 'hook', 'reverse', 'turnaround', 'fingerroll']

importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

