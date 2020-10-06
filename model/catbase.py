#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('/home/mengyuan/huawei')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from preprocess import under_sample,tools,feature_process

import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import gc
import numpy as np
from scipy.stats import entropy
from gensim.models import Word2Vec
from sklearn.metrics import *


train_df = under_sample.under_sample_train_data()

test_df = pd.read_csv('/home/mengyuan/huawei/data/test_data_B.csv',sep='|')

train_df,test_df=feature_process.generate_features(train_df,test_df)


#线下数据集的切分
X_train = train_df[train_df["pt_d"]<=6].copy()
y_train = X_train["label"].astype('int32')
X_valid = train_df[train_df["pt_d"]>6]
y_valid = X_valid["label"].astype('int32')


#筛选特征
drop_fea = ['pt_d','label','communication_onlinerate','index']
feature= [x for x in X_train.columns if x not in drop_fea]
print(len(feature))
print(feature)


#线下验证
cate_fea = []
clf = CatBoostClassifier(iterations=10000, depth=6,learning_rate=0.1, loss_function='Logloss',cat_features=cate_fea
                        ,verbose=True,eval_metric='AUC',task_type='GPU',metric_period=50)
clf.fit(
    X_train[feature], y_train.astype('int32'),
    eval_set=[(X_valid[feature],y_valid.astype('int32'))],
    early_stopping_rounds=200,
    verbose=True,
    use_best_model=True
)

# y_predprob = clf.predict_proba(X_valid[feature])[:, 1]

# y_pre = clf.predict(test_df[feature])[:, 1]
# auc_score =roc_auc_score(y_valid, y_predprob)
# print("AUC Score (Valid): %f" % auc_score)



# #查看模型的特征重要性
# import matplotlib.pyplot as plt
# from matplotlib import cm
# score = pd.DataFrame()
# score['fea_name'] = clf.feature_names_
# score['fea']=clf.feature_importances_
# score = score.sort_values(['fea'], ascending=False)
# score.to_csv('huawei/feature_weight.csv')
# temp = pd.DataFrame()
# temp = score[:60]
# color = cm.jet(temp['fea']/temp['fea'].max())
# plt.figure(figsize=(25, 17))
# plt.barh(temp['fea_name'],temp['fea'],height =0.8,color=color,alpha=0.8)
# # plt.show()
# plt.savefig('huawei/feature_weight.jpg')



#线上提交的模型训练
clf1 = CatBoostClassifier(iterations=clf.best_iteration_, depth=6,learning_rate=0.1, loss_function='Logloss'
                        ,eval_metric='AUC',task_type='GPU',metric_period=50)
clf1.fit(
    train_df[feature], train_df['label'].astype('int32'),
    verbose=True
)
y_pre = clf1.predict_proba(test_df[feature])[:, 1]

import pickle
with open('/data/mengyuan/huawei/model/catbase.pkl', 'wb') as f:
    pickle.dump(clf1, f)
    print('save catebase model to /data/mengyuan/huawei/model/catbase.pkl !!')


res = pd.DataFrame()
res['id'] = test_df['id'].astype('int32')
res['probability'] = y_pre
res.to_csv('/data/mengyuan/huawei/ensemble/submission_catbase.csv',index = False)

