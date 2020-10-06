#!/usr/bin/env python
# coding: utf-8

# 此代码的tensorflow的版本为1.3.1,deepctr的版本为0.7.4，这个无法在tensorflow2以及最新的deepctr上运行
import sys
sys.path.append('/home/mengyuan/huawei')
from preprocess import under_sample,tools,feature_process

import pandas as pd
from tqdm import tqdm
import gc
import numpy as np
from sklearn.metrics import *
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.models import *
from deepctr.feature_column import  SparseFeat, DenseFeat, get_feature_names
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,2" #指定GPU



#相关参数
weights_path = './fibinet_base.h5'
learning_rate = 1e-3
batch = 8192*4
n_epoch=100
embedding_dim = 8 #embedding维度一般来说越大越好，但是维度越大跑起来越慢


train_df = pd.read_csv('/data/mengyuan/train_data.csv',sep='|')
train_df = train_df.sample(frac=0.5).reset_index(drop=True)
test_df = pd.read_csv('/data/mengyuan/test_data_A.csv',sep='|')
df = pd.concat([train_df,test_df],axis=0)
test_id = test_df['id'].copy().reset_index(drop=True)

df=df.replace([np.inf, -np.inf],0)
df=df.fillna(0)


#处理类别特征
cate_cols = ['city_rank','creat_type_cd','dev_id','device_size','gender','indu_name','inter_type_cd','residence','slot_id','net_type','task_id','adv_id','adv_prim_id','age','app_first_class','app_second_class','career','city','consume_purchase','uid','dev_id','tags']
for f in tqdm(cate_cols):
    map_dict = dict(zip(df[f].unique(), range(df[f].nunique())))
    df[f] = df[f].map(map_dict).fillna(-1).astype('int32')
    df[f + '_count'] = df[f].map(df[f].value_counts())
df = tools.reduce_mem(df)



drop_fea = ['pt_d','label','communication_onlinerate','index','uid','dev_id']
feature= [x for x in df.columns if x not in drop_fea]
print(len(feature))
print(feature)

sparse_features = cate_cols
dense_features = [x for x in df.columns if x not in drop_fea+cate_cols] #这里的dense_feature可以把树模型的特征加进来
print('sparse_feature: {}'.format(sparse_features))
print('dense_feature: {}'.format(dense_features))



#对dense_features进行归一化
mms = MinMaxScaler(feature_range=(0,1))
df[dense_features] = mms.fit_transform(df[dense_features])



fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].nunique(),embedding_dim=embedding_dim)
                       for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                      for feat in dense_features]



test_df = df[df["pt_d"]==8].copy().reset_index()
train_df = df[df["pt_d"]<8].reset_index()
del df
gc.collect()



dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
online_train_model_input = {name:train_df[name].values for name in feature_names}
online_test_model_input = {name:test_df[name].values for name in feature_names}



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
    return tf.compat.v1.py_func(roc_auc_score, (y_true, y_pred), tf.double)



plateau = ReduceLROnPlateau(monitor="val_auroc", verbose=1, mode='max', factor=0.3, patience=5)
early_stopping = EarlyStopping(monitor='val_auroc', patience=9, mode='max')
checkpoint = ModelCheckpoint(weights_path,
                             monitor='val_auroc',
                             verbose=0,
                             mode='max',
                             save_best_only=True)

model = FiBiNET(linear_feature_columns,dnn_feature_columns,task='binary',dnn_dropout=0.1,dnn_hidden_units=(512, 128),)

opt = Adam(lr=learning_rate)
model.compile(optimizer=opt,
              #loss='binary_crossentropy',
              loss = multi_category_focal_loss2(alpha=0.1, gamma=2),
              metrics=[auroc], )

history = model.fit(online_train_model_input, train_df['label'].values,
                    validation_split=0.2,
                    callbacks=[early_stopping,plateau,checkpoint],shuffle=True,
                    batch_size=batch, epochs=n_epoch)



model.load_weights(weights_path)
y_pre = model.predict(online_test_model_input, batch_size=batch)



res = pd.DataFrame()
res['id'] = test_id
res['probability'] = y_pre


res.to_csv('./fibinet_base.csv',index=False)

