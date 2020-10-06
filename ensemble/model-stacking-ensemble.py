#!/usr/bin/env python
# coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import sys
sys.path.append('/home/mengyuan/huawei')
from preprocess import under_sample,tools,feature_process

import numpy as np
import pandas as pd
import datetime
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import log_loss,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge, Lasso, LarsCV, RidgeCV, Lars
import warnings
import random

import datetime
# from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import catboost as cb
from catboost import CatBoostClassifier
import scipy
from sklearn.cluster import DBSCAN
# from pandas.api.types import is_numeric_dtype


from keras import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint


warnings.filterwarnings('ignore')
np.random.seed(1)
random.seed(1)

import pickle

# 
# ## Modeling
# 
# Here we use [out of fold stacking ensemble]
# (https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/).
# The architecture is as followed:
# 
# **Layer 1**:
# * 2 lightgbm
# * 1 xgboost
# * 1 catboost
# * 1 dense neural network
# 
# **Layer 2**:
# * Lasso regression
# * Ridge regression
# 


# train = under_sample.under_sample_train_data()
#
# test = pd.read_csv('/home/mengyuan/huawei/data/test_data_B.csv',sep='|')
#
# train,test=feature_process.generate_features(train,test)
#
# ######存储结果节点(代替上面数据载入)###################################################################################
# with open('/data/mengyuan/huawei/ensemble/{}.pkl'.format('train'), 'wb') as file:
#     pickle.dump(train, file)
# with open('/data/mengyuan/huawei/ensemble/{}.pkl'.format('test'), 'wb') as file:
#     pickle.dump(test, file)
######################################################################################################
####读取节点##############
with open('/data/mengyuan/huawei/ensemble/{}.pkl'.format('train'), 'rb') as file:
    train=pickle.load(file)
with open('/data/mengyuan/huawei/ensemble/{}.pkl'.format('test'), 'rb') as file:
    test=pickle.load(file)
print('成功读入train、test的pickle。。。。')
#####################################################################################################

target = train['label'].astype('int32')
FEATS_EXCLUDED = ['pt_d','label','communication_onlinerate','index']
features = [c for c in train.columns if c not in FEATS_EXCLUDED]



# ### First layer
# #### Tree-based model



# List of model to use
ITERATIONS = 20000
best_params = {
    'device': 'gpu',
    # 'gpu_platform_id': 1,
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
lgb1 = lgb.LGBMClassifier(**best_params)
# xgb1 = xgb.XGBRegressor(learning_rate= 0.5843131085630997,
#                         booster= 'gbtree',
#                         alpha = 1.0239095745311145e-08,
#                         boosting= 'gbdt',
#                         num_leaves= 31,
#                         colsample_bytree= 0.5406862297709868,
#                         subsample= 0.9594952525792275,
#                         max_depth= 5,
#                         reg_lambda= 3.143121436123109,
#                         eta= 2.9162386740282797e-07,
#                         gamma= 6.015464829655147e-07,
#                         grow_policy= 'depthwise',
#                        random_state=2018,
#                         )
xgb1=xgb.XGBClassifier(objective='binary:logistic',eval_metric='auc',iterations=ITERATIONS,device ='gpu',gpu_device_id=0,learning_rate=0.18128719397842585,alpha=0.4525288269053296,colsample_bytree=0.6829173136955252,subsample=0.9527989030259758,max_depth=12,reg_lambda=0.8818041076646994,eta=3.931530871551245e-05,gamma=0.0003499513067203738,grow_policy='lossguide',sample_type= 'weighted',normalize_type='forest',rate_drop=0.006050806976220067,skip_drop=6.399689613706687e-05)
cate_fea = []
cb1 = CatBoostClassifier(iterations=ITERATIONS, depth=6,learning_rate=0.1, loss_function='Logloss',cat_features=cate_fea
                        ,verbose=True,eval_metric='AUC',counter_calc_method='Full',task_type='GPU',metric_period=50)


N_FOLDS=7
layer1_models = [cb1,xgb1,lgb1 ]#, ada1]
layer1_names = [ 'catboost1','xgboost1','lightgbm1']#, 'adaboost1']


# In[15]:


oof_train = np.zeros(shape=(len(train),len(layer1_models)))
oof_test = np.zeros(shape=(len(test),len(layer1_models)))

# Recording results
layer1_score = []
feature_importance = []


# In[17]:

###################################################################################################
# for i in range(len(layer1_models)):
#     feature_importance_df = pd.DataFrame()
#     print('\n')
#     name = layer1_names[i]
#     model = layer1_models[i]
#     folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=2021+i)
#     print('Training %s' %name)
#     for fold_, (trn_idx, val_idx) in enumerate(folds.split(train[features], train['label'].astype('int32'))):
#         print('Fold no %i/%i'%(fold_+1,N_FOLDS))
#         trn_data = train[features].iloc[trn_idx]
#         trn_label = train['label'].iloc[trn_idx]
#         val_data = train[features].iloc[val_idx]
#         val_label = train['label'].iloc[val_idx]
#         if 'ada' in name:
#             model.fit(X=trn_data, y=trn_label.astype('int32'))
#         else:
#             model.fit(X=trn_data, y=trn_label.astype('int32'),
#                      eval_set=[(val_data, val_label.astype('int32'))],
#                      verbose=True,
#                      early_stopping_rounds=200)
#
#         oof_train[val_idx,i] = model.predict(val_data)
#         oof_test[:,i] += model.predict(test[features])/N_FOLDS
#
#         fold_importance_df = pd.DataFrame()
#         fold_importance_df["feature"] = features
#         fold_importance_df["importance"] = model.feature_importances_
#         fold_importance_df["fold"] = fold_ + 1
#         feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
#
#     score = log_loss(oof_train[:,i], target)
#     layer1_score.append(score)
#     feature_importance.append(feature_importance_df)
#     print('Training CV score-logloss: %.5f' %score)

######存储结果节点(代替上面for循环运行结果)###################################################################################
# with open('/data/mengyuan/huawei/ensemble/{}.pkl'.format('layer1_score'), 'wb') as file:
#     pickle.dump(layer1_score, file)
# with open('/data/mengyuan/huawei/ensemble/{}.pkl'.format('feature_importance'), 'wb') as file:
#     pickle.dump(feature_importance, file)
# with open('/data/mengyuan/huawei/ensemble/{}.pkl'.format('oof_train'), 'wb') as file:
#     pickle.dump(oof_train, file)
# with open('/data/mengyuan/huawei/ensemble/{}.pkl'.format('oof_test'), 'wb') as file:
#     pickle.dump(oof_test, file)
######################################################################################################
####读取节点##############
with open('/data/mengyuan/huawei/ensemble/{}.pkl'.format('layer1_score'), 'rb') as file:
    layer1_score=pickle.load(file)
with open('/data/mengyuan/huawei/ensemble/{}.pkl'.format('feature_importance'), 'rb') as file:
    feature_importance=pickle.load(file)
with open('/data/mengyuan/huawei/ensemble/{}.pkl'.format('oof_train'), 'rb') as file:
    oof_train=pickle.load(file)
with open('/data/mengyuan/huawei/ensemble/{}.pkl'.format('oof_test'), 'rb') as file:
    oof_test=pickle.load(file)
print('成功读入layer1_score、feature_importance、oof_train、oof_test的pickle。。。。')
#####################################################################################################

# Exchange two models of cv-bagging###########

oof_preds=np.loadtxt('/home/mengyuan/huawei/model/lgb_oof_preds.csv',delimiter=',')
sub_preds=np.loadtxt('/home/mengyuan/huawei/model/lgb_sub_preds.csv',delimiter=',')

score = score = log_loss(target, oof_preds)
print('Training CV score-logloss for LGB: %.5f' %score)
layer1_names.append('LGB')
layer1_score.append(score)

oof_preds = oof_preds[:, np.newaxis]
sub_preds = sub_preds[:, np.newaxis]
oof_train = np.hstack((oof_train, oof_preds))
oof_test = np.hstack((oof_test, sub_preds))



oof_preds=np.loadtxt('/home/mengyuan/huawei/model/xgb_oof_preds.csv',delimiter=',')
sub_preds=np.loadtxt('/home/mengyuan/huawei/model/xgb_sub_preds.csv',delimiter=',')

score = score = log_loss(target, oof_preds)
print('Training CV score-logloss for XGB: %.5f' %score)
layer1_names.append('XGB')
layer1_score.append(score)

oof_preds = oof_preds[:, np.newaxis]
sub_preds = sub_preds[:, np.newaxis]
oof_train = np.hstack((oof_train, oof_preds))
oof_test = np.hstack((oof_test, sub_preds))



print('layer1_names:'+str(layer1_names))


# #### Neural network

# In[21]:


# Preparation
import pandas as pd
from tqdm import tqdm
import gc
import numpy as np
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.models import *
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.optimizers import Adam, RMSprop, SGD



import keras.backend as K
import tensorflow.compat.v1 as tf
# from sklearn.preprocessing import StandardScaler, Imputer
K.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))



# train_df = pd.read_csv('/data/mengyuan/train_data.csv',sep='|')
# train_df = train_df.sample(frac=0.5).reset_index(drop=True)
train_df =under_sample.under_sample_train_data()
test_df = pd.read_csv('/data/mengyuan/test_data_B.csv',sep='|')
df = pd.concat([train_df,test_df],axis=0)
test_id = test_df['id'].copy().reset_index(drop=True)

df=df.replace([np.inf, -np.inf],0)
df=df.fillna(0)

test_df = df[df["pt_d"]==8].copy().reset_index()
train_df = df[df["pt_d"]<8].reset_index()
del df
gc.collect()

FEATS_EXCLUDED = ['pt_d','label','communication_onlinerate','index']
features = [c for c in train_df.columns if c not in FEATS_EXCLUDED]

# 相关参数
weights_path = '/data/mengyuan/huawei/ensemble/fibinet_base.h5'
learning_rate = 1e-3
batch = 8192 * 4
n_epoch = 30
embedding_dim = 8  # embedding维度一般来说越大越好，但是维度越大跑起来越慢

###########################################
print('处理FiBiNET的train data###############')
df = train_df
df = df.replace([np.inf, -np.inf], 0)
df = df.fillna(0)

# 处理类别特征
cate_cols = ['city_rank', 'creat_type_cd', 'dev_id', 'device_size', 'gender', 'indu_name', 'inter_type_cd',
             'residence', 'slot_id', 'net_type', 'task_id', 'adv_id', 'adv_prim_id', 'age', 'app_first_class',
             'app_second_class', 'career', 'city', 'consume_purchase', 'uid', 'dev_id', 'tags']
for f in tqdm(cate_cols):
    map_dict = dict(zip(df[f].unique(), range(df[f].nunique())))
    df[f] = df[f].map(map_dict).fillna(-1).astype('int32')
    df[f + '_count'] = df[f].map(df[f].value_counts())
df = tools.reduce_mem(df)

drop_fea = ['pt_d', 'label', 'communication_onlinerate', 'index', 'uid', 'dev_id']
feature = [x for x in df.columns if x not in drop_fea]
print(len(feature))
print(feature)

sparse_features = cate_cols
dense_features = [x for x in df.columns if x not in drop_fea + cate_cols]  # 这里的dense_feature可以把树模型的特征加进来
print('sparse_feature: {}'.format(sparse_features))
print('dense_feature: {}'.format(dense_features))

# 对dense_features进行归一化
mms = MinMaxScaler(feature_range=(0, 1))
df[dense_features] = mms.fit_transform(df[dense_features])

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].nunique(), embedding_dim=embedding_dim)
                          for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                        for feat in dense_features]
traindf = df
del df
gc.collect()
###########################################
print('处理FiBiNET的test data###############')
df = test_df
df = df.replace([np.inf, -np.inf], 0)
df = df.fillna(0)

# 处理类别特征
cate_cols = ['city_rank', 'creat_type_cd', 'dev_id', 'device_size', 'gender', 'indu_name', 'inter_type_cd',
             'residence', 'slot_id', 'net_type', 'task_id', 'adv_id', 'adv_prim_id', 'age', 'app_first_class',
             'app_second_class', 'career', 'city', 'consume_purchase', 'uid', 'dev_id', 'tags']
for f in tqdm(cate_cols):
    map_dict = dict(zip(df[f].unique(), range(df[f].nunique())))
    df[f] = df[f].map(map_dict).fillna(-1).astype('int32')
    df[f + '_count'] = df[f].map(df[f].value_counts())
df = tools.reduce_mem(df)

drop_fea = ['pt_d', 'label', 'communication_onlinerate', 'index', 'uid', 'dev_id']
feature = [x for x in df.columns if x not in drop_fea]
print(len(feature))
print(feature)

sparse_features = cate_cols
dense_features = [x for x in df.columns if x not in drop_fea + cate_cols]  # 这里的dense_feature可以把树模型的特征加进来
print('sparse_feature: {}'.format(sparse_features))
print('dense_feature: {}'.format(dense_features))

# 对dense_features进行归一化
mms = MinMaxScaler(feature_range=(0, 1))
df[dense_features] = mms.fit_transform(df[dense_features])

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].nunique(), embedding_dim=embedding_dim)
                          for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, )
                                                                        for feat in dense_features]
testdf = df
del df
gc.collect()
##########################################
dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

#################################################################
oof_test_nn = np.zeros(shape=(len(test_df),1))
oof_train_nn = np.zeros(shape=(len(train_df),1))



def nn_model(trn_data,val_data):
    online_train_model_input = {name: trn_data[name].values for name in feature_names}
    online_val_model_input = {name: val_data[name].values for name in feature_names}

    def multi_category_focal_loss2(gamma=2., alpha=.25):
        """
        focal loss for multi category of multi label problem
        适用于多分类或多标签问题的focal loss
        alpha控制真值y_true为1/0时的权重
            1的权重为alpha, 0的权重为1-alpha
        当你的模型欠拟合，学习存在困难时，可以尝试适用本函数作为loss
        当模型过于激进(无论何时总是倾向于预测出1),尝试将alpha调小
        当模型过于惰性(无论何时总是倾向于预测出0,或是某一个固定的常数,说明没有学到有效特征)
            尝试将alpha调大,鼓励模型进行预测出1。
        Usage:
         model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
        """
        epsilon = 1.e-7
        gamma = float(gamma)
        alpha = tf.constant(alpha, dtype=tf.float32)

        def multi_category_focal_loss2_fixed(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

            alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
            y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
            ce = -tf.math.log(y_t)
            weight = tf.pow(tf.subtract(1., y_t), gamma)
            fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
            loss = tf.reduce_mean(fl)
            return loss

        return multi_category_focal_loss2_fixed

    def auroc(y_true, y_pred):
        return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

    plateau = ReduceLROnPlateau(monitor="val_auroc", verbose=1, mode='max', factor=0.3, patience=5)
    early_stopping = EarlyStopping(monitor='val_auroc', patience=9, mode='max')
    checkpoint = ModelCheckpoint(weights_path,
                                 monitor='val_auroc',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)

    model = FiBiNET(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=0.1,
                    dnn_hidden_units=(512, 128), )

    opt = Adam(lr=learning_rate)
    model.compile(optimizer=opt,
                  # loss='binary_crossentropy',
                  loss=multi_category_focal_loss2(alpha=0.1, gamma=2),
                  metrics=[auroc], )
    history = model.fit(online_train_model_input, trn_data['label'].values,
                        validation_data=(online_val_model_input, val_data['label'].values),
                        callbacks=[early_stopping, plateau, checkpoint], shuffle=True,
                        batch_size=batch, epochs=n_epoch)

    model.load_weights(weights_path)
    return model



## nn训练
folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=2019)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(traindf[feature_names].values, traindf['label'].values)):
    print('Fold no %i/%i'%(fold_+1,N_FOLDS))
    trn_data = traindf.iloc[trn_idx]
    val_data = traindf.iloc[val_idx]
    # model = nn_model(trn_data.shape[1])
    # hist = model.fit(trn_data,trn_label,
    #                  validation_data = (val_data, val_label),
    #                  epochs=EPOCHS,
    #                  batch_size=512,
    #                  verbose=True,
    #                  callbacks=[early_stop])
    model = nn_model(trn_data,val_data)

    online_val_model_input = {name: val_data[name].values for name in feature_names}
    online_test_model_input = {name: testdf[name].values for name in feature_names}

    oof_train_nn[val_idx,0] = model.predict(online_val_model_input)[:,0]
    oof_test_nn[:,0] += model.predict(online_test_model_input)[:,0]/N_FOLDS


######存储nn结果节点(代替上面nn训练结果)###################################################################################
with open('/data/mengyuan/huawei/ensemble/{}.pkl'.format('oof_train_nn'), 'wb') as file:
    pickle.dump(oof_train_nn, file)
with open('/data/mengyuan/huawei/ensemble/{}.pkl'.format('oof_test_nn'), 'wb') as file:
    pickle.dump(oof_test_nn, file)
print('成功存下oof_train_nn、oof_test_nn的pickle。。。。')
######################################################################################################
# ####读取节点##############
# with open('/data/mengyuan/huawei/ensemble/{}.pkl'.format('oof_train_nn'), 'rb') as file:
#     oof_train_nn=pickle.load(file)
# with open('/data/mengyuan/huawei/ensemble/{}.pkl'.format('oof_test_nn'), 'rb') as file:
#     oof_test_nn=pickle.load(file)
# print('成功读入oof_train_nn、oof_test_nn的pickle。。。。')
#####################################################################################################

score_nn = mean_squared_error(oof_train_nn[:,0], traindf['label'])
print('Training CV score-mean_squared_error for neural network: %.5f' %score_nn)
layer1_names.append('neural_net')
layer1_score.append(score_nn)

oof_train = np.hstack((oof_train, oof_train_nn))
oof_test = np.hstack((oof_test, oof_test_nn))


# #### Layer 1 summary


# Print first layer result
print('Print first layer result')
layer1 = pd.DataFrame()
layer1['models'] = layer1_names
layer1['CV_score_logloss'] = layer1_score

print('layer1:'+str(layer1))




layer1_corr = pd.DataFrame()
for i in range(len(layer1_names)):
    layer1_corr[layer1_names[i]] = oof_train[:,i]
layer1_corr['label'] = train['label']
colormap = plt.cm.RdBu
plt.figure(figsize=(12,12))
sns.heatmap(layer1_corr.astype(float).corr(), linewidths=0.1, vmax=1.0, vmin=-1., square=True, cmap=colormap, linecolor='white', annot=True)
plt.title('Pair-wise correlation')
plt.savefig('/data/mengyuan/huawei/ensemble/Pair-wise_correlation.jpg')

# ### Second layer

# Setup the model
lr=LogisticRegression(solver='saga')
# ridge = Ridge(alpha=0.5)#, fit_intercept=False)
# lasso = Lasso(alpha=0.5)
# lars = Lars(fit_intercept=False, positive=True)
# layer2_models = [lars]#[ridge]# lasso]
# layer2_names = ['Lars']#['ridge'] #, 'lasso']
# #params_grid = {'alpha':[0.05,0.1,0.4,1.0]}
layer2_models = [lr]
layer2_names = ['LogisticRegression']


# Setup to record result
train_pred = np.zeros(len(train))
test_pred = np.zeros(len(test))

layer2 = pd.DataFrame()
layer2['models'] = layer2_names
layer2_score = []


# Taking average
print("Taking average oof_train#############")
train_pred = np.mean(oof_train, axis=1)
print('Training score-mean_squared_error: %.5f' %mean_squared_error(train_pred, train['label']))
print("Taking average oof_test#############")
test_pred = np.mean(oof_test, axis=1)
print('Training score-mean_squared_error: %.5f' %mean_squared_error(test_pred, test['label']))

# For regression
print("For regression################################################")
for i in range(len(layer2_models)):
    print('\n')
    name = layer2_names[i]
    model = layer2_models[i]
    print('Training %s' %name)
    model.fit(oof_train, target)
    score = mean_squared_error(model.predict_proba(oof_train), target)
    train_pred += model.predict_proba(oof_train)/len(layer2_models)
    test_pred += model.predict_proba(oof_test)/len(layer2_models)
    layer2_score.append(score)
    print('Training score-do_regressor: %.5f' % score)

print('Print second layer result')
layer2['CV score'] = layer2_score

print('layer2:'+str(layer2))



# ### Submission

# In[33]:


#sub_df = pd.DataFrame({"card_id":test["card_id"].values})
res1 = pd.DataFrame()
res1['id'] = test_df['id'].astype('int32')
res1['probability'] = test_pred
res1.to_csv('/data/mengyuan/huawei/ensemble/submission1.csv',index = False)


sample_submission = pd.read_csv('/home/mengyuan/huawei/data/submission3_0.796365.csv')
res2 = pd.DataFrame()
res2['id'] = test_df['id'].astype('int32')
res2['probability'] = test_pred*0.5 + sample_submission['probability']*0.5
res2.to_csv('/data/mengyuan/huawei/ensemble/submission2.csv',index = False)

res3 = pd.DataFrame()
res3['id'] = test_df['id'].astype('int32')
res3['probability'] = test_pred*0.3 + sample_submission['probability']*0.7
res3.to_csv('/data/mengyuan/huawei/ensemble/submission3.csv',index = False)

res4 = pd.DataFrame()
res4['id'] = test_df['id'].astype('int32')
res4['probability'] = test_pred*0.2 + sample_submission['probability']*0.8
res4.to_csv('/data/mengyuan/huawei/ensemble/submission4.csv',index = False)


print('submission save!')
# In[ ]:





# In[34]:


# plt.figure(figsize=(8,5))
# plt.scatter(range(sub_df.shape[0]), np.sort(sub_df['target'].values))
# plt.xlabel('index', fontsize=12)
# plt.ylabel('Loyalty Score', fontsize=12)
# plt.title('Loyalty score after scaling')
# plt.show()


# In[ ]:

layer2_coef = pd.DataFrame()
layer2_coef['Name'] = layer1_names
layer2_coef['Coefficient'] = model.coef_
#layer2_coef['Coefficient'] = coef

print('layer2_coef:'+str(layer2_coef))




#np.sum(model.coef_)


plt.figure(figsize=(8,5))
plt.scatter(range(len(test_pred)), np.sort(test_pred))
plt.xlabel('index', fontsize=12)
plt.ylabel('Loyalty Score', fontsize=12)
plt.title('Loyalty score before scaling')
plt.savefig('/data/mengyuan/huawei/ensemble/Loyalty_score_before_scaling.jpg')


# In[32]:


# Refit to the target
# train_scaler = StandardScaler()
#train_scaler.fit(target.values.reshape(-1,1))
#test_pred = train_scaler.inverse_transform(test_pred.reshape(-1,1))





# In[ ]:





# In[ ]:




