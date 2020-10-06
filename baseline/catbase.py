#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# In[ ]:


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


train_df = pd.read_csv('/data/mengyuan/train_data.csv',sep='|')
train_df = train_df.sample(frac = 0.5).reset_index(drop=True)    #根据你的内存来决定
print('label:'+str(train_df['label'].unique()))

test_df = pd.read_csv('/data/mengyuan/test_data_B.csv',sep='|')
df = pd.concat([train_df,test_df],axis=0)


# In[ ]:


##########################cate feature#######################
cate_cols = ['slot_id','net_type','task_id','adv_id','adv_prim_id','age','app_first_class','app_second_class','career','city','consume_purchase','uid','dev_id','tags']
for f in tqdm(cate_cols):
    map_dict = dict(zip(df[f].unique(), range(df[f].nunique())))
    df[f + '_count'] = df[f].map(df[f].value_counts())
df = reduce_mem(df)


# In[ ]:


##########################groupby feature#######################
def group_fea(df,key,target):
    tmp = df.groupby(key, as_index=False)[target].agg({
        key+target + '_nunique': 'nunique',
    }).reset_index()
    del tmp['index']
    print("**************************{}**************************".format(target))
    return tmp

feature_key = ['uid','age','career','net_type']
feature_target = ['task_id','adv_id','dev_id','slot_id','spread_app_id','indu_name']

for key in tqdm(feature_key):
    for target in feature_target:
        tmp = group_fea(df,key,target)
        df = df.merge(tmp,on=key,how='left')


# In[ ]:


test_df = df[df["pt_d"]==8].copy().reset_index()
train_df = df[df["pt_d"]<8].reset_index()
del df
gc.collect()


# In[ ]:


#统计做了groupby特征的特征
group_list = []
for s in train_df.columns:
    if '_nunique' in s:
        group_list.append(s)
print(group_list)


# In[ ]:


##########################target_enc feature#######################
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
enc_list = group_list + ['net_type','task_id','adv_id','adv_prim_id','age','app_first_class','app_second_class','career','city','consume_purchase','uid','uid_count','dev_id','tags','slot_id']
for f in tqdm(enc_list):
    train_df[f + '_target_enc'] = 0
    test_df[f + '_target_enc'] = 0
    for i, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        trn_x = train_df[[f, 'label']].iloc[trn_idx].reset_index(drop=True)
        val_x = train_df[[f]].iloc[val_idx].reset_index(drop=True)
        enc_df = trn_x.groupby(f, as_index=False)['label'].agg({f + '_target_enc': 'mean'})
        val_x = val_x.merge(enc_df, on=f, how='left')
        test_x = test_df[[f]].merge(enc_df, on=f, how='left')
        val_x[f + '_target_enc'] = val_x[f + '_target_enc'].fillna(train_df['label'].mean())
        test_x[f + '_target_enc'] = test_x[f + '_target_enc'].fillna(train_df['label'].mean())
        train_df.loc[val_idx, f + '_target_enc'] = val_x[f + '_target_enc'].values
        test_df[f + '_target_enc'] += test_x[f + '_target_enc'].values / skf.n_splits
        


# In[ ]:


#线下数据集的切分
X_train = train_df[train_df["pt_d"]<=6].copy()
y_train = X_train["label"].astype('int32')
X_valid = train_df[train_df["pt_d"]>6]
y_valid = X_valid["label"].astype('int32')


# In[ ]:


#筛选特征
drop_fea = ['pt_d','label','communication_onlinerate','index']
feature= [x for x in X_train.columns if x not in drop_fea]
print(len(feature))
print(feature)


# In[ ]:


#线下验证
cate_fea = []
clf = CatBoostClassifier(iterations=10000, depth=6,learning_rate=0.1, loss_function='Logloss',cat_features=cate_fea
                        ,verbose=True,eval_metric='AUC',counter_calc_method='Full',task_type='GPU',metric_period=50)
clf.fit(
    X_train[feature], y_train.astype('int32'),
    eval_set=[(X_valid[feature],y_valid.astype('int32'))],
    early_stopping_rounds=200,
    verbose=True,
    use_best_model=True,
)
y_predprob = clf.predict_proba(X_valid[feature])[:, 1] 

y_pre = clf.predict_proba(test_df[feature])[:, 1]  
auc_score =roc_auc_score(y_valid, y_predprob)
print("AUC Score (Valid): %f" % auc_score) 


# In[ ]:


#查看模型的特征重要性
import matplotlib.pyplot as plt 
from matplotlib import cm
score = pd.DataFrame()
score['fea_name'] = clf.feature_names_
score['fea']=clf.feature_importances_
score = score.sort_values(['fea'], ascending=False)
temp = pd.DataFrame()
temp = score[:60]
color = cm.jet(temp['fea']/temp['fea'].max())
plt.figure(figsize=(10, 15))
plt.barh(temp['fea_name'],temp['fea'],height =0.8,color=color,alpha=0.8)
plt.show()


# In[ ]:


#线上提交的模型训练
clf1 = CatBoostClassifier(iterations=clf.best_iteration_, depth=6,learning_rate=0.1, loss_function='Logloss'
                        ,eval_metric='AUC',counter_calc_method='Full',task_type='GPU',metric_period=50)
clf1.fit(
    train_df[feature], train_df['label'].astype('int32'),
    verbose=True,
    use_best_model=True,
)
y_pre = clf1.predict_proba(test_df[feature])[:, 1]    

res = pd.DataFrame()
res['id'] = test_df['id'].astype('int32')
res['probability'] = y_pre
res.to_csv('/data/mengyuan/huawei/ensemble/baseline_catbase_{}.csv'.format(auc_score),index = False)

