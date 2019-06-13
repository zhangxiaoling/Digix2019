# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 22:03:52 2019

@author: 11747
"""

import numpy as np
import pandas as pd
import time
import gc
import datetime
from sklearn import preprocessing
encoder=preprocessing.LabelEncoder()

# ad_info
data_ad=pd.read_csv('ad_info.csv',names=['adId','billId','primId','creativeType','intertype','spreadAppId'])
data_ad.drop('creativeType',axis=1,inplace=True)
data_ad.drop('intertype',axis=1,inplace=True)

data_test=pd.read_csv('test_20190518.csv',names=['id','uId','adId','operTime','siteId','slotId','contentId','netType'])
sub=data_test[['id']]
data_test.drop('operTime',axis=1,inplace=True)
data_test.drop('netType',axis=1,inplace=True)
data_test.drop('id',axis=1,inplace=True)
data_test.drop('uId',axis=1,inplace=True)

data_train=pd.read_csv('train_20190518.csv',nrows=50000000,skiprows=50000000,names=['label','uId','adId','operTime','siteId','slotId','contentId','netType'])

label=data_train[['label']]
data_train.drop('uId',axis=1,inplace=True)
data_train.drop('operTime',axis=1,inplace=True)
data_train.drop('netType',axis=1,inplace=True)

data_test=pd.merge(data_test,data_ad,on=['adId'],how='left')

data_train=pd.merge(data_train,data_ad,on=['adId'],how='left')

data=pd.concat([data_train,data_test])

del data_train
del data_test
gc.collect()

for feat in ['adId','billId','primId','spreadAppId','contentId','siteId','slotId']:
    res=pd.DataFrame()
    temp=data[[feat,'label']]
    count=temp.groupby([feat]).apply(lambda x:x['label'].count()).reset_index(name=feat+'all')
    count.fillna(value=0, inplace=True)
    res=res.append(count,ignore_index=True)
    print('over')
    data=pd.merge(data,res,how='left',on=[feat])
    

data['spreadAppId']=data['spreadAppId'].fillna(-1)
data['spreadAppIdall']=data['spreadAppIdall'].fillna(0)
data['contentId']=data['contentId'].fillna(-1)
data['contentIdall']=data['contentIdall'].fillna(0)

col_encoder=['adId','billId','contentId','primId','siteId','slotId','spreadAppId']
for feat in col_encoder:
    encoder.fit(data[feat])
    data[feat]=encoder.transform(data[feat])
    
train=data[:50000000]
test=data[50000000:]

train.to_csv('train.csv',header=False,index=False)
test.to_csv('test.csv',header=False,index=False)



