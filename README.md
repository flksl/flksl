import pandas as pd
import numpy as np
import os
import xgboost as xgb

kernel_log_data_path = 'memory_sample_kernel_log_round1_a_train.csv'
failure_tag_data_path = 'memory_sample_failure_tag_round1_a_train.csv'

#读一下数据
org_kernel_log = pd.read_csv(kernel_log_data_path)#('memory_sample_kernel_log_round1_b_test.csv')

#看一下有哪些列
org_kernel_log.keys()

#聚合到5分钟粒度
org_kernel_log['collect_time'] = pd.to_datetime(org_kernel_log['collect_time']).dt.ceil("5min")

#按sum整理一下
group_min = org_kernel_log.groupby(['serial_number','collect_time'],as_index=False).agg('sum')

#读取tag文件
failure_tag = pd.read_csv(failure_tag_data_path)

#链接为一个表
failure_tag['failure_time']= pd.to_datetime(failure_tag['failure_time'])
merged_data = pd.merge(group_min,failure_tag[['serial_number','failure_time']],how='left',on=['serial_number'])
merged_data.keys()

#计算下时间间隔
merged_data['failure_dis']=(merged_data['failure_time'] - merged_data['collect_time']).dt.total_seconds()

#做图看一下数据分布
import matplotlib.pyplot as plt

f= plt.plot(merged_data['failure_dis'])
plt.show()
from IPython.display import display
display(f)

# 去掉大于1200的数据
remove_id = []
for sn, tmp_df in merged_data.groupby('serial_number', as_index=False):
    if np.min(tmp_df['failure_dis'].values) > 1200:
        remove_id.extend(list(tmp_df.index))
org_size = merged_data.shape[0]
merged_data = merged_data.drop(remove_id).reset_index(drop=True)
new_size = merged_data.shape[0]
print("filter: %d -> %d" % (org_size, new_size))

# 以300、600、1200为分割点做多分类lable
label = np.zeros(merged_data.shape[0], dtype=int)
label[merged_data['failure_dis'] < 300] = 1
label[(merged_data['failure_dis'] >= 300) & (merged_data['failure_dis'] < 600)] = 2
label[(merged_data['failure_dis'] >= 600) & (merged_data['failure_dis'] < 1200)] = 3

merged_data['failure_tag'] = label
merged_data.drop('failure_dis', axis=1, inplace=True)

#看一下最终的数据
merged_data['failure_tag'].head(5)

#看一下各类数据分布
print(list(merged_data['failure_tag']).count(0))
print(list(merged_data['failure_tag']).count(1))
print(list(merged_data['failure_tag']).count(2))
print(list(merged_data['failure_tag']).count(3))

feature_data = merged_data.drop(['serial_number', 'collect_time','manufacturer','vendor','failure_time'], axis=1)
# 负样本下采样
sample_0 = feature_data[feature_data['failure_tag']==0].sample(frac=0.1)
sample = sample_0.append(feature_data[feature_data['failure_tag']!=0])

sample.keys()

#用xgb做多分类
xlf_keraddmce = xgb.XGBClassifier(max_depth=8, learning_rate=0.05, n_estimators=200, reg_alpha=0.005, n_jobs=8,
                                      importance_type='total_cover',
                                      random_state=2021, use_label_encoder=False, eval_metric='logloss')
                                      #tree_method='gpu_hist')



xlf_keraddmce.fit(sample.iloc[:,:-1],sample['failure_tag'])

# 测试数据
group_data_test = pd.read_csv('memory_sample_kernel_log_k12_round1_a_test.csv')
group_data_test['collect_time'] = pd.to_datetime(group_data_test['collect_time']).dt.ceil('5min')
group_data_test = group_data_test.groupby(['serial_number','collect_time'],as_index=False).agg('sum')

group_min_sn_test = pd.DataFrame(group_data_test[['serial_number','collect_time']])
group_min_test = group_data_test.drop(['serial_number', 'collect_time','manufacturer','vendor'], axis=1)

# 模型预测
res = xlf_keraddmce.predict(group_min_test)
group_min_sn_test['predict']=res

# 保存结果
group_min_sn_test=group_min_sn_test[group_min_sn_test['predict']==1]
group_min_sn_res = group_min_sn_test.drop('predict',axis=1)
group_min_sn_res.to_csv('memory_predit_res_svm.csv', header=False, index=False)

group_min_sn_res.head(5)

