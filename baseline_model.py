# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 22:09:55 2019

@author: 11747
"""

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb
import numpy as np
import pandas as pd
import time
import gc
import datetime
from sklearn import preprocessing

train=pd.read_csv('train.csv',names=['adId','billId','contentId','label','primId','siteId','slotId','spreadAppId','adIdall','billIdall','primIdall','spreadAppIdall','contentIdall','siteIdall','slotIdall'])
test=pd.read_csv('test.csv',names=['adId','billId','contentId','label','primId','siteId','slotId','spreadAppId','adIdall','billIdall','primIdall','spreadAppIdall','contentIdall','siteIdall','slotIdall']

y_train=train.loc[:,'label']
id=range(1,1000001)
train.drop(['label'],axis=1,inplace=True)
test.drop(['label'],axis=1,inplace=True)

X_loc_train=train.values
y_loc_train=y_train.values
X_loc_test=test.values

model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.05, n_estimators=2000,

                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,

                           min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,

                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True)
model = lgb.LGBMClassifier()

lgb_model = model.fit(X_loc_train, y_loc_train,
                          eval_metric='logloss')
test_pred= lgb_model.predict_proba(X_loc_test, num_iteration=lgb_model.best_iteration_)[:, 1]

print('pred:',test_pred)
dataset=list(zip(id,test_pred))
df=pd.DataFrame(data=dataset,columns=['id','probability'])
df.to_csv('submission.csv',index=False,header=True)