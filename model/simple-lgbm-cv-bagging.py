#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('/home/mengyuan/huawei')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from preprocess import under_sample,tools,feature_process

import gc
import time
from datetime import datetime as dt

import numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

import warnings
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

import lightgbm as lgb
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

from contextlib import contextmanager



##训练数据和测试数据
train_df = under_sample.under_sample_train_data()
test_df = pd.read_csv('/home/mengyuan/huawei/data/test_data_B.csv',sep='|')

train_df,test_df=feature_process.generate_features(train_df,test_df)

df = pd.concat([train_df, test_df], axis=0)#合并
#拆分
# test_df = df[df["pt_d"]==8].copy().reset_index()
# train_df = df[df["pt_d"]<8].reset_index()
# del df
# gc.collect()


FEATS_EXCLUDED = ['pt_d','label','communication_onlinerate','index']
CATE_COLS = ['slot_id', 'net_type', 'task_id', 'adv_id', 'adv_prim_id', 'age', 'app_first_class', 'app_second_class',
                 'career', 'city', 'consume_purchase', 'uid', 'dev_id', 'tags']


def modeling_lgbm_cross_validation(params, X, y, nr_folds=5, verbose=0):
    clfs = list()
    oof_preds = np.zeros(X.shape[0])
    # Split data with kfold
    kfolds = KFold(n_splits=nr_folds, shuffle=True, random_state=42)
    for n_fold, (trn_idx, val_idx) in enumerate(kfolds.split(X, y)):
        if verbose:
            print('no {} of {} folds'.format(n_fold, nr_folds))

        X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=verbose, eval_metric='auc',
            early_stopping_rounds=500
        )

        clfs.append(model)
        oof_preds[val_idx] = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:,1]

        del X_train, y_train, X_valid, y_valid
        gc.collect()
    np.savetxt("/home/mengyuan/huawei/model/lgb_oof_preds.csv", oof_preds, delimiter=",")
    score = mean_squared_error(y, oof_preds) ** .5
    return clfs, score


def predict_cross_validation(test, clfs, ntree_limit=None):
    sub_preds = np.zeros(test.shape[0])
    test_preds = np.zeros(test.shape[0])
    for i, model in enumerate(clfs, 1):

        num_tree = 10000
        if not ntree_limit:
            ntree_limit = num_tree

        if isinstance(model, lgb.sklearn.LGBMClassifier):
            if model.best_iteration_:
                num_tree = min(ntree_limit, model.best_iteration_)
            print('test_preds.shape:'+str(test_preds.shape))
            print('model.predict_proba(test, num_iteration=num_tree):'+str(model.predict_proba(test, num_iteration=num_tree).shape))
            print('model.predict_proba(test, num_iteration=num_tree)[:,1]:' + str(model.predict_proba(test,num_iteration=num_tree)[:,1].shape))

            test_preds = model.predict_proba(test, num_iteration=num_tree)[:,1]

        if isinstance(model, xgb.sklearn.XGBClassifier):
            num_tree = min(ntree_limit, model.best_ntree_limit)
            test_preds = model.predict_proba(test, ntree_limit=num_tree)[:,1]

        sub_preds += test_preds

    sub_preds = sub_preds / len(clfs)
    np.savetxt("/home/mengyuan/huawei/model/lgb_sub_preds.csv", sub_preds, delimiter=",")
    ret = pd.Series(sub_preds, index=test.index)
    ret.index.name = test.index.name
    return ret



               
# "Run LightGBM with kfold"
train_features = [c for c in train_df.columns if c not in FEATS_EXCLUDED]
'''train_features = pd.read_csv('boruta-feature-ranking2.csv')
train_features =train_features.loc[train_features['rank'] ==1]['features'].tolist()
feature = ['feature_1', 'feature_2' ,'feature_3']
train_features.extend(feature)
print(train_features)'''
best_params = {
    'device': 'gpu',
    'gpu_platform_id': 1,
    'gpu_device_id': 0,
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'n_jobs': 4, 'max_depth': 7,
    'n_estimators': 2000,
    'subsample_freq': 2,
    'subsample_for_bin': 200000,
    'min_data_per_group': 100,
    'max_cat_to_onehot': 4,
    'cat_l2': 10.0,
    'cat_smooth': 10.0,
    'max_cat_threshold': 32,
    'metric_freq': 10,
    'verbosity': -1,
    'metric': 'auc',
    'colsample_bytree': 0.5,
    'learning_rate': 0.01,
    'min_child_samples': 80,
    'min_child_weight': 100.0,
    'min_split_gain': 1e-06,
    'num_leaves': 47,
    'reg_alpha': 10.0,
    'reg_lambda': 10.0,
    'subsample': 0.9}

# modeling
nr_folds = 7
best_params.update({'n_estimators': 20000})
clfs = list()
score = 0
# clfs, score = modeling_lgbm_cross_validation(best_params,
#                                             train_df[train_features],
#                                             train_df['label'],
#                                             nr_folds,
#                                             verbose=50)

import pickle
# with open('/data/mengyuan/huawei/model/lgbm_cv_bagging.pkl', 'wb') as f:
#     pickle.dump(clfs, f)
#     print('save lgbm_cv_bagging model to /data/mengyuan/huawei/model/lgbm_cv_bagging.pkl !!')
with open('/data/mengyuan/huawei/model/lgbm_cv_bagging.pkl', 'rb') as f:
    clfs=pickle.load(f)
    print('读取 lgbm_cv_bagging model 从 /data/mengyuan/huawei/model/lgbm_cv_bagging.pkl !!')


# save to
# file_template = '{score:.6f}_{model_key}_cv{fold}_{timestamp}'
# file_stem = file_template.format(
#     score=score,
#     model_key='LGBM',
#     fold=nr_folds,
#     timestamp=dt.now().strftime('%Y-%m-%d-%H-%M'))

# filename = '/home/mengyuan/huawei/model/submission_{}.csv'.format(file_stem)
filename='/data/mengyuan/huawei/ensemble/submission_lgbm_cv_bagging.csv'
print('save to {}'.format(filename))
# subm = predict_cross_validation(test_df[train_features], clfs)
# subm = subm.to_frame('target')
# subm.to_csv(filename, index=True)
y_pre = predict_cross_validation(test_df[train_features], clfs)

res = pd.DataFrame()
res['id'] = test_df['id'].astype('int32')
res['probability'] = y_pre
res.to_csv(filename,index = False)
print('succeed to save！')
# #############################################################
# import pandas as pd
#
# subm= pd.read_csv(filename,header=None)
# sub=subm.loc[1:]
# res = pd.DataFrame()
# res['probability'] = sub[1]
# res.insert(0, 'id', [i for i in range(1,1000001)])
# # res['id'] = test_df['id'].astype('int32')
# res.to_csv('/home/mengyuan/huawei/model/baseline_lgbm.csv',index = False)