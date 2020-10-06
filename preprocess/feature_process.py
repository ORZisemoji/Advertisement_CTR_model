
import sys
sys.path.append('/home/mengyuan/huawei')
from preprocess import under_sample,tools

import pandas as pd
from tqdm import tqdm
import gc

def generate_features(train_df,test_df):

    df = pd.concat([train_df, test_df], axis=0)

    ##########################cate feature#######################
    cate_cols = ['slot_id', 'net_type', 'task_id', 'adv_id', 'adv_prim_id', 'age', 'app_first_class', 'app_second_class',
                 'career', 'city', 'consume_purchase', 'uid', 'dev_id', 'tags']
    for f in tqdm(cate_cols):
        map_dict = dict(zip(df[f].unique(), range(df[f].nunique())))
        df[f + '_count'] = df[f].map(df[f].value_counts())
    df = tools.reduce_mem(df)

    ##########################groupby feature#######################
    def group_fea(df, key, target):
        tmp = df.groupby(key, as_index=False)[target].agg({
            key + target + '_nunique': 'nunique',
        }).reset_index()
        del tmp['index']
        print("**************************{}**************************".format(target))
        return tmp

    feature_key = ['uid', 'age', 'career', 'net_type']
    feature_target = ['task_id', 'adv_id', 'dev_id', 'slot_id', 'spread_app_id', 'indu_name']
    for key in tqdm(feature_key):
        for target in feature_target:
            tmp = group_fea(df, key, target)
            df = df.merge(tmp, on=key, how='left')

    test_df = df[df["pt_d"] == 9].copy().reset_index()
    train_df = df[df["pt_d"] < 9].reset_index()
    del df
    gc.collect()

    # 统计做了groupby特征的特征
    group_list = []
    for s in train_df.columns:
        if '_nunique' in s:
            group_list.append(s)
    print(group_list)


    ##########################target_enc feature#######################
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
    enc_list = group_list + ['net_type', 'task_id', 'adv_id', 'adv_prim_id', 'age', 'app_first_class', 'app_second_class',
                             'career', 'city', 'consume_purchase', 'uid', 'uid_count', 'dev_id', 'tags', 'slot_id']
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

    return train_df,test_df


