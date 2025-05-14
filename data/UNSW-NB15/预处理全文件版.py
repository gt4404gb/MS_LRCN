import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder

class_mapping = {
    'Normal': 0,
    'Analysis': 1,
    'Backdoor': 2,
    'DoS': 3,
    'Exploits': 4,
    'Fuzzers': 5,
    'Generic': 6,
    'Reconnaissance': 7,
    'Shellcode': 8,
    'Worms': 9
}


dfs = []
for i in range(1,5):
    path = 'UNSW-NB15_{}.csv'  # There are 4 input csv files
    dfs.append(pd.read_csv(path.format(i), header = None))
combined_data = pd.concat(dfs).reset_index(drop=True)  # Concat all to a single df

combined_data.head()

dataset_columns = pd.read_csv('NUSW-NB15_features.csv',encoding='ISO-8859-1')
dataset_columns.info()

combined_data.columns = dataset_columns['Name']
combined_data.info()

combined_data.head()

combined_data['Label'].value_counts()

combined_data['attack_cat'].isnull().sum()

combined_data['attack_cat'] = combined_data['attack_cat'].fillna(value='normal').apply(lambda x: x.strip().lower())

combined_data['attack_cat'].value_counts()

combined_data['attack_cat'] = combined_data['attack_cat'].replace('backdoors','backdoor', regex=True).apply(lambda x: x.strip().lower())

combined_data['attack_cat'].value_counts()

combined_data.isnull().sum()

combined_data['ct_flw_http_mthd'] = combined_data['ct_flw_http_mthd'].fillna(value=0)

combined_data['is_ftp_login'].value_counts()

combined_data['is_ftp_login'] = combined_data['is_ftp_login'].fillna(value=0)

combined_data['is_ftp_login'].value_counts()

combined_data['is_ftp_login'] = np.where(combined_data['is_ftp_login']>1, 1, combined_data['is_ftp_login'])

combined_data['is_ftp_login'].value_counts()

combined_data['service'].value_counts()

#combined_data['service'] = combined_data['servie'].replace(to_replace='-', value='None')
combined_data['service'] = combined_data['service'].apply(lambda x:"None" if x=='-' else x)

combined_data['service'].value_counts()

combined_data['ct_ftp_cmd'].unique()

combined_data['ct_ftp_cmd'] = combined_data['ct_ftp_cmd'].replace(to_replace=' ', value=0).astype(int)

combined_data['ct_ftp_cmd'].unique()

combined_data[['service','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','attack_cat','Label']]

combined_data['attack_cat'].nunique()

combined_data.drop(columns=['srcip','sport','dstip','dsport','Label'],inplace=True)

combined_data.info()

print(combined_data.shape)

# 保存整体结果为CSV文件
#combined_data.to_csv("kaggle_UNSW_NB15_full.csv", index=False)

#分割训练集与测试集，分别进行处理
train, test = train_test_split(combined_data,test_size=0.5,random_state=16)

x_train, y_train = train.drop(columns=['attack_cat']), train[['attack_cat']]
x_test, y_test = test.drop(columns=['attack_cat']), test[['attack_cat']]

print(x_train.shape, y_train.shape)

cat_col = ['proto', 'service', 'state']
num_col = list(set(x_train.columns) - set(cat_col))

#标准化和归一化
minmaxscaler = MinMaxScaler()
scaler = StandardScaler()
scaler = scaler.fit(x_train[num_col])
x_train[num_col] = scaler.transform(x_train[num_col])
x_test[num_col] = scaler.transform(x_test[num_col])

x_train.isnull().sum()

x_train[num_col] = minmaxscaler.fit_transform(x_train[num_col])
x_test[num_col] = minmaxscaler.transform(x_test[num_col])



ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), cat_col)], remainder='passthrough')
x_train = np.array(ct.fit_transform(x_train))
x_test = np.array(ct.transform(x_test))

train_attacks = y_train['attack_cat'].unique()
# Get unique elements and their counts
unique_values, counts = np.unique(y_train, return_counts=True)

# Print the unique values and their corresponding counts
for value, count in zip(unique_values, counts):
    print(f"Value: {value}, Count: {count}")

# Get unique elements and their counts
unique_values, counts = np.unique(y_test, return_counts=True)

# Print the unique values and their corresponding counts
for value, count in zip(unique_values, counts):
    print(f"Value: {value}, Count: {count}")

#ct1 = ColumnTransformer(transformers=[('encoder', LabelEncoder(categories=[train_attacks],sparse=False), ['attack_cat'])], remainder='passthrough')
ct1 = LabelEncoder()

y_train = np.array(ct1.fit_transform(y_train))
y_test = np.array(ct1.transform(y_test))

print(x_train.shape,y_train.shape,y_test.shape)

# 你也可以直接查看每个数字编码对应的标签
for i in range(len(ct1.classes_)):
    print(f"Number {i} corresponds to label '{ct1.classes_[i]}'")
# 为了沿着列方向拼接，需要将y_train调整形状为 (n_samples, 1)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# 现在使用numpy.concatenate进行拼接
train = np.concatenate([x_train, y_train], axis=1)
test = np.concatenate([x_test, y_test], axis=1)

# 最后将结果转换为DataFrame
train = pd.DataFrame(train)
test = pd.DataFrame(test)

#train = pd.concat([x_train, y_train], axis=1)
#test = pd.concat([x_test, y_test], axis=1)
#val = pd.concat([x_val, y_val], axis=1)

train.to_csv("kaggle_UNSW_NB15_full_training.csv", index=False)
test.to_csv("kaggle_UNSW_NB15_full_testing.csv", index=False)

