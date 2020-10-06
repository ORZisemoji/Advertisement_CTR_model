#!/usr/bin/env python
# coding: utf-8

# 此代码的tensorflow的版本为1.3.1,deepctr的版本为0.7.4，这个无法在tensorflow2以及最新的deepctr上运行

# In[1]:


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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr.models import *
from deepctr.feature_column import  SparseFeat, DenseFeat, get_feature_names
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import tensorflow as tf
import os

from keras import backend as K


# In[2]:


os.environ['CUDA_VISIBLE_DEVICES'] = '0， 1， 3' #指定GPU


# In[3]:


def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                           100*(start_mem-end_mem)/start_mem,
                                                                                                           (time.time()-starttime)/60))
    return df


# In[ ]:


#相关参数
weights_path = './fibinet_base.h5'
learning_rate = 1e-3
batch = 8192*4
n_epoch=100
embedding_dim = 8 #embedding维度一般来说越大越好，但是维度越大跑起来越慢

#
# # In[4]:
#
#
# get_ipython().run_cell_magic('time', '', "train_df = pd.read_csv('/data/mengyuan/train_data.csv',sep='|')\ntrain_df = train_df.sample(frac=0.5).reset_index(drop=True)\ntest_df = pd.read_csv('/data/mengyuan/test_data_A.csv',sep='|')\ndf = pd.concat([train_df,test_df],axis=0)\ntest_id = test_df['id'].copy().reset_index(drop=True)")
#
train_df = pd.read_csv('/home/mengyuan/huawei/data/train_data.csv',sep='|')
print('label:'+str(train_df['label'].unique()))
#看一下正负样本的具体数据量情况
y=train_df["label"]
print(y.value_counts())
print("-------------------------")
print(y.value_counts(normalize=True))
# 0    40461645
# # 1     1445488
# # Name: label, dtype: int64
# # -------------------------
# # 0    0.965507
# # 1    0.034493
# # Name: label, dtype: float64
# train_df = train_df.sample(frac = 0.1).reset_index(drop=True)    #根据你的内存来决定
###################0907  下采样
df1=train_df[train_df["label"]==1]#正样本部分
df0=train_df[train_df["label"]==0]#负样本部分
#对负样本按0.05的比例进行下采样
df2=df0.sample(frac=0.05).reset_index(drop=True)
#将下采样后的正样本与负样本进行组合
train_df=pd.concat([df2,df1],axis=0)
#看一下正负样本的具体数据量情况
y=train_df["label"]
print(y.value_counts())
print("-------------------------")
print(y.value_counts(normalize=True))
# 0    2023082
# 1    1445488
# Name: label, dtype: int64
# -------------------------
# 0    0.583261
# 1    0.416739
# Name: label, dtype: float64
test_df = pd.read_csv('/home/mengyuan/huawei/data/test_data_A.csv',sep='|')
df = pd.concat([train_df,test_df],axis=0)
test_id = test_df['id'].copy().reset_index(drop=True)
#
# # In[7]:
#
#
# get_ipython().run_cell_magic('time', '', 'df=df.replace([np.inf, -np.inf],0)\ndf=df.fillna(0)')
df=df.replace([np.inf, -np.inf],0)
df=df.fillna(0)


# In[15]:


#处理类别特征
cate_cols = ['city_rank','creat_type_cd','dev_id','device_size','gender','indu_name','inter_type_cd','residence','slot_id','net_type','task_id','adv_id','adv_prim_id','age','app_first_class','app_second_class','career','city','consume_purchase','uid','dev_id','tags']

for f in tqdm(cate_cols):
    map_dict = dict(zip(df[f].unique(), range(df[f].nunique())))
    df[f] = df[f].map(map_dict).fillna(-1).astype('int32')
    df[f + '_count'] = df[f].map(df[f].value_counts())
df = reduce_mem(df)


# In[16]:


drop_fea = ['pt_d','label','communication_onlinerate','index','uid','dev_id']
feature= [x for x in df.columns if x not in drop_fea]
print(len(feature))
print(feature)


# In[17]:


sparse_features = cate_cols
dense_features = [x for x in df.columns if x not in drop_fea+cate_cols] #这里的dense_feature可以把树模型的特征加进来
print('sparse_feature: {}'.format(sparse_features))
print('dense_feature: {}'.format(dense_features))


# In[18]:


#对dense_features进行归一化
mms = MinMaxScaler(feature_range=(0,1))
df[dense_features] = mms.fit_transform(df[dense_features])


# In[19]:


fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=df[feat].nunique(),embedding_dim=embedding_dim)
                       for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                      for feat in dense_features]


# In[20]:


test_df = df[df["pt_d"]==8].copy().reset_index()
train_df = df[df["pt_d"]<8].reset_index()
from sklearn.utils import shuffle
train_df = shuffle(train_df)
del df
gc.collect()


# In[24]:


dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
online_train_model_input = {name:train_df[name].values for name in feature_names}
online_test_model_input = {name:test_df[name].values for name in feature_names}


# In[23]:


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
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)
    # return tf.compat.v1.py_func(roc_auc_score, (y_true, y_pred), tf.double)



# In[ ]:


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


# In[ ]:


model.load_weights(weights_path)
y_pre = model.predict(online_test_model_input, batch_size=batch)


# In[ ]:


res = pd.DataFrame()
res['id'] = test_id
res['probability'] = y_pre


# In[ ]:


res.to_csv('/home/mengyuan/huawei/fibinet_base.csv',index=False)

