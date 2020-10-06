#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('/home/mengyuan/huawei')
from preprocess import under_sample,tools,feature_process

import datetime
import gc
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import warnings

from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

import optuna


warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


FEATS_EXCLUDED = ['pt_d','label','communication_onlinerate','index']





def objective(trial):

        num_folds = 11
        train_x, train_y = train_df[feats], train_df['label']
        #data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
        '''dtrain = lgb.Dataset(train_x, label=train_y)'''

        lgbm_train = lgb.Dataset(train_x,
                                 train_y,
                                  free_raw_data=False
                                  )

        params = {'objective': 'binary',
                  'metric': 'auc',
                  # 'verbosity': -1,
                  "learning_rate": trial.suggest_uniform('learning_rate', 0.001, 1),
                  # 'device': 'gpu',
                  # 'gpu_platform_id': 1,
                  # 'gpu_device_id': 0,
                  'num_thread' : 1,
                  'sparse_threshold' : 1,
                  'seed': 2779,
                  #'boosting_type': trial.suggest_categorical('boosting', ['gbdt',  'goss']),
                  'num_leaves': trial.suggest_int('num_leaves', 16, 200),
                  'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.001, 1),
                  'subsample': trial.suggest_uniform('subsample', 0.001, 1),
                  'max_depth': trial.suggest_int('max_depth', 3, 20),
                  'reg_alpha': trial.suggest_uniform('reg_alpha', 0, 10),
                   'reg_lambda': trial.suggest_uniform('reg_lambda', 0, 10),
                  'min_split_gain': trial.suggest_uniform('min_split_gain', 0, 10),
                  'min_child_weight': trial.suggest_uniform('min_child_weight', 0, 45),
                  'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 16, 64),

                  'min_child_samples' : trial.suggest_int('min_child_samples', 1, 200),
                  #'num_iterations': trial.suggest_uniform('num_iterations', 1, 5000),
                  'feature_fraction' : trial.suggest_uniform('feature_fraction', 0.001, 1),
                  #'random_state': trial.suggest_int('random_state', 1, 5000),
                  #'max_bin' :  trial.suggest_int('random_state', 1, 256)
                  }

        '''if params['boosting_type'] == 'dart':
            params['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
            params['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
        if params['boosting_type'] == 'goss':
            params['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
            params['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - params['top_rate'])'''

        #folds = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=47)
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)


        clf = lgb.cv(
                        params,
                        lgbm_train,
                        metrics=['auc'],
                        nfold=num_folds,
                        folds=folds.split(train_df[feats], train_df['label']),
                        num_boost_round=10000,
                        early_stopping_rounds= 200,
                        verbose_eval=100,
                        seed=47

                         )
        print("交叉验证AUC:{},参数:{}".format(max(clf['auc-mean']), params))
        gc.collect()
        return max(clf['auc-mean'])





train_df = under_sample.under_sample_train_data()
test_df = pd.read_csv('/home/mengyuan/huawei/data/test_data_A.csv',sep='|')
train_df,test_df=feature_process.generate_features(train_df,test_df)

feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print('Number of finished trials: {}'.format(len(study.trials)))

print('Best trial:')
trial = study.best_trial

print('  Value: {}'.format(trial.value))

print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

hist_df = study.trials_dataframe()
hist_df.to_csv("/home/mengyuan/huawei/getparams/optuna_result_lgbm.csv")





