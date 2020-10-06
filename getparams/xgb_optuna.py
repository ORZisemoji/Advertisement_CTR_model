#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('/home/mengyuan/huawei')
from preprocess import under_sample,tools,feature_process

import datetime
import gc
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

import xgboost as xgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

import optuna

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


FEATS_EXCLUDED = ['pt_d','label','communication_onlinerate','index']




def objective(trial):
    
        data = train_df[feats]
        target = train_df['label']
        train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dtest = xgb.DMatrix(test_x, label=test_y)
        

        param = {'objective': 'binary:logistic',
                  'eval_metric': 'auc',
                  "learning_rate": trial.suggest_uniform('learning_rate', 0.001, 1),

                  'booster': trial.suggest_categorical('booster', ['gbtree','dart']),
                  'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
                  'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
                  
                  #'gpu_id': 0,
                  #'tree_method': 'gpu_hist',
                  #'max_bin': 16,
                  #'updater' : 'grow_gpu_hist',

                
                  'seed': 326,
                  # 'boosting_type': trial.suggest_categorical('boosting', ['gbdt',  'goss']),
                  # 'num_leaves': trial.suggest_int('num_leaves', 16, 64),
                  'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.2, 1.0),
                  'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
                  'max_depth': trial.suggest_int('max_depth', 5, 20),
                  'reg_alpha': trial.suggest_uniform('reg_alpha', 0, 10),
                  'reg_lambda': trial.suggest_uniform('reg_lambda', 0, 10),
                  'min_split_gain': trial.suggest_uniform('min_split_gain', 0, 10),
                  'min_child_weight': trial.suggest_uniform('min_child_weight', 0, 45),
                  'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 16, 64)
                  }


        
        if param['booster'] == 'gbtree' or param['booster'] == 'dart':
            param['max_depth'] = trial.suggest_int('max_depth', 1, 9)
            param['eta'] = trial.suggest_loguniform('eta', 1e-8, 1.0)
            param['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
            param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
        if param['booster'] == 'dart':
            param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
            param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            param['rate_drop'] = trial.suggest_loguniform('rate_drop', 1e-8, 1.0)
            param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)

        # Add a callback for pruning.
        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")

        gbm = xgb.train(param, dtrain,evals=[(dtest, "validation")], callbacks=[pruning_callback])
        preds = gbm.predict(dtest)
        pred_labels = np.rint(preds)
        accuracy = sklearn.metrics.accuracy_score(test_y, pred_labels)
        
        return accuracy



train_df = under_sample.under_sample_train_data()
test_df = pd.read_csv('/home/mengyuan/huawei/data/test_data_A.csv',sep='|')
train_df,test_df=feature_process.generate_features(train_df,test_df)

feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),direction="maximize")
study.optimize(objective, n_trials=100)

print('Number of finished trials: {}'.format(len(study.trials)))

print('Best trial:')
trial = study.best_trial

print('  Value: {}'.format(trial.value))

print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

hist_df = study.trials_dataframe()
hist_df.to_csv("/home/mengyuan/huawei/getparams/optuna_result_xgbm.csv")
