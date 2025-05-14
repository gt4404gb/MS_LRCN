import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

def writeData(path):
    return pd.read_csv(path)

# 按行合并多个Dataframe数据
def mergeData():
    monday = writeData("Monday-WorkingHours.pcap_ISCX.csv")

    # 剔除第一行属性特征名称
    monday = monday.drop([0])
    friday1 = writeData("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    friday1 = friday1.drop([0])
    friday2 = writeData("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    friday2 = friday2.drop([0])
    friday3 = writeData("Friday-WorkingHours-Morning.pcap_ISCX.csv")
    friday3 = friday3.drop([0])
    thursday1 = writeData("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
    thursday1 = thursday1.drop([0])
    thursday2 = writeData("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
    thursday2 = thursday2.drop([0])
    tuesday = writeData("Tuesday-WorkingHours.pcap_ISCX.csv")
    tuesday = tuesday.drop([0])
    wednesday = writeData("Wednesday-workingHours.pcap_ISCX.csv")
    wednesday = wednesday.drop([0])
    frame = [monday, tuesday,wednesday, thursday1, thursday2,friday1, friday2, friday3]

    # 合并数据
    raw_pd = pd.concat(frame)
    result = clearDirtyData(raw_pd)
    #result = check(result)  确定处理后无Nan值，无需check了
    result = replaceLabel(result)
    result = standard(result)
    return result


# 清除CIC-IDS数据集中的脏数据，第一行特征名称和含有Nan、Infiniti等数据的行数
def clearDirtyData(df):
    # 找出含有实际NaN值或'Nan'/'inf'字符串的行索引
    nan_inf_rows = df[(df.iloc[:, 14].isin(["Nan", "inf"]) | pd.isna(df.iloc[:, 14])) |
                      (df.iloc[:, 15].isin(["Infinity", "inf"]) | pd.isna(df.iloc[:, 15]))].index

    # 直接从DataFrame中删除这些行
    df_cleaned = df.drop(index=nan_inf_rows)

    # 删除包含无穷大值的行
    df_cleaned = df_cleaned[~df_cleaned.isin([np.inf, -np.inf]).any(axis=1)]

    return df_cleaned

def check(data):
    # 检查是否存在 NaN 值
    nan_check = data.isnull().sum().sum()
    # 检查是否存在无穷大值
    inf_check = data.isin([np.Inf, -np.Inf]).sum().sum()
    if nan_check > 0 or inf_check > 0:
        print("11111")
    return data

def replaceLabel(df):
    # 假设df是你的pandas Da   taFrame
    labels, unique = pd.factorize(df[' Label'])

    mapping = {i: unique[i] for i in range(len(unique))}
    print(mapping)
    # {0: 'BENIGN', 1: 'FTP-Patator', 2: 'SSH-Patator', 3: 'DoS slowloris', 4: 'DoS Slowhttptest', 5: 'DoS Hulk', 6: 'DoS GoldenEye', 7: 'Heartbleed', 8: 'Infiltration', 9: 'Web Attack � Brute Force', 10: 'Web Attack � XSS', 11: 'Web Attack � Sql Injection', 12: 'DDoS', 13: 'PortScan', 14: 'Bot'}

    # 现在labels数组包含了转换后的数值标签
    df[' Label'] = labels
    return df


def standard(df):
    # 提取最后一列为y
    y = df.iloc[:, -1]
    # 除去最后一列的数据作为特征
    x = df.iloc[:, :-1]

    # 初始化标准化器和归一化器
    scaler = MinMaxScaler()
    transfer = StandardScaler()

    # 对特征数据进行标准化处理
    x_scaled = transfer.fit_transform(x)

    # 对标准化后的数据进行归一化处理
    x_normalized = scaler.fit_transform(x_scaled)

    # 将处理后的特征数据与标签列合并
    x_train_encoded = pd.DataFrame(x_normalized, columns=df.columns[:-1])
    x_train_encoded = pd.concat([x_train_encoded, y.reset_index(drop=True)], axis=1)

    return x_train_encoded

clean_data = mergeData()
file = 'cicids2017.csv'
clean_data.to_csv(file, index=False, header=False)
print("数据预处理完成")