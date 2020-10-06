#encoding:utf-8
import pandas as pd

def under_sample_train_data(data_path='/data/mengyuan/train_data.csv'):#旧服务器：/home/mengyuan/huawei/data/train_data.csv
	train_df = pd.read_csv(data_path, sep='|')
	print('label:' + str(train_df['label'].unique()))
	# 看一下正负样本的具体数据量情况
	y = train_df["label"]
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
	df1 = train_df[train_df["label"] == 1]  # 正样本部分
	df0 = train_df[train_df["label"] == 0]  # 负样本部分
	# 对负样本按0.05的比例进行下采样
	df2 = df0.sample(frac=0.05).reset_index(drop=True)
	# 将下采样后的正样本与负样本进行组合
	train_df = pd.concat([df2, df1], axis=0)
	# 看一下正负样本的具体数据量情况
	y = train_df["label"]
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
	return train_df

