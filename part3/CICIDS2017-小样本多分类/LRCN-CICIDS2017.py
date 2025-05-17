#!/usr/bin/env python
# coding: utf-8

# In[1]:
print("测试内容：LRCN，CICIIDS2017小样本测试")
from torch.nn import init, Parameter
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import umap
import src.retnet as tf
import model.LRCN as retnet

#——————————配置调用设备————————————
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 数据放到gpu上还是cpu上
print("device",dev)
#——————————配置调用设备————————————

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
channel_size=1
hidden_dim = 256
latent_dim = 16
learn_rate = 0.001
batch_size = 1024
class_number = 15  #分类类别
epochs = 1


# In[3]:


#————————————————————————————数据集加载————————————————————————————————
x = pd.read_csv("../../data/CICIDS2017/cicids2017.csv", low_memory=False)
num_cols = x.shape[1] #获取列数
#提取出最后一列为y
y = x.pop(x.columns[-1]).values

# 将读取的数据转化为np格式方便后续训练
x_train = np.array(y, dtype=np.float32) # 将数据转换为float32类型
y_train = np.array(y, dtype=np.int64) # 将数据转换为int64类型

#设置使用数据集比例
proportion = 0.01
# 随机划分，比例为70%的训练数据和30%的测试数据
x_train, x_test, y_train, y_test = train_test_split(x.values, y, train_size=80*proportion, test_size=20*proportion, random_state=42)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# 转换为 PyTorch 张量并移动到设备
x_train = torch.from_numpy(x_train).float().to(dev)
y_train = torch.from_numpy(y_train).long().to(dev)
x_test = torch.from_numpy(x_test).float().to(dev)
y_test = torch.from_numpy(y_test).long().to(dev)

# 定义数据加载器
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#————————————————————————————数据集加载————————————————————————————————


# In[4]:


#model = LSTMClassifier(input_size=x_train.shape[1], hidden_size=hidden_dim, output_size=class_number).to(dev)
model = retnet.TFClassifier(input_size=x_train.shape[1], hidden_size=hidden_dim, output_size=class_number).to(dev)
optimizer = optim.Adam(model.parameters(), lr=learn_rate)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)


# In[5]:


#训练模型
model.train()
losses = []
train_start_time = time.time()

for epoch in range(epochs):
    start_time = time.time()
    total_loss = 0
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        h,y_hat = model(x_batch)
        loss = retnet.tf_loss(y_batch, y_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss/len(train_loader))
    print('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, epochs, losses[-1]))
    end_time = time.time()
    print("Cost time:",end_time - start_time)

train_end_time = time.time()
print("Train Cost time:",train_start_time - train_end_time)

# In[6]:


#并行调用
print("并行调用模型测试")
y_true = []
y_pred_total = []

model.eval()
x_total = torch.tensor([]).cpu()
z_total = torch.tensor([]).cpu()
quantized_total = torch.tensor([]).cpu()
recon_total = torch.tensor([]).cpu()
y_total = torch.tensor([]).cpu()
y_hat_total = torch.tensor([]).cpu()

start_time = time.time()
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        h,y_hat = model(x_batch)
        y_true += y_batch.tolist()
        y_pred = np.argmax(y_hat.cpu(), axis=1)
        y_pred_total += y_pred.tolist()
        x_total = torch.cat([x_total.cpu(), x_batch.cpu()], dim=0)
        z_total = torch.cat([z_total.cpu(), h.cpu()], dim=0)
        quantized_total = torch.cat([quantized_total.cpu(), y_hat.cpu()], dim=0)
        y_total = torch.cat([y_total.cpu(), y_batch.cpu()], dim=0)
        y_hat_total = torch.cat([y_hat_total.cpu(), y_pred.cpu()], dim=0)

end_time = time.time()
print("Cost time:", end_time - start_time)

acc = accuracy_score(y_true, y_pred_total)
f1_weighted = f1_score(y_true, y_pred_total, average='weighted')
f1_macro = f1_score(y_true, y_pred_total, average='macro')
f1_micro = f1_score(y_true, y_pred_total, average='micro')
precision_macro = precision_score(y_true, y_pred_total, average='macro')
recall_macro = recall_score(y_true, y_pred_total, average='macro')
precision_micro = precision_score(y_true, y_pred_total, average='micro')
recall_micro = recall_score(y_true, y_pred_total, average='micro')

# 打印结果
print('Test Accuracy: {:.4f}'.format(acc))
print('Test Weighted F1 Score: {:.4f}'.format(f1_weighted))
print('Macro F1 Score: {:.4f}'.format(f1_macro))
print('Micro F1 Score: {:.4f}'.format(f1_micro))
print('Macro Precision: {:.4f}, Macro Recall: {:.4f}'.format(precision_macro, recall_macro))
print('Micro Precision: {:.4f}, Micro Recall: {:.4f}'.format(precision_micro, recall_micro))

# 绘制Loss图
# 假设 losses 是你在训练过程中收集的loss列表
# 示例：losses = [0.7, 0.6, 0.5, 0.4, 0.3]
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 二值化预测和真实标签
# 示例中的classes参数应根据实际类别数量进行调整
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14])
y_pred_bin = label_binarize(y_pred_total, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14])

# 分类报告
print(classification_report(y_true, y_pred_total))

# 计算每个类别的AUC及总的AUC
auc_scores = roc_auc_score(y_true_bin, y_pred_bin, average=None)
auc_macro = roc_auc_score(y_true_bin, y_pred_bin, average='macro')
auc_micro = roc_auc_score(y_true_bin, y_pred_bin, average='micro')

# 打印AUC结果
print("AUC Scores by class:", auc_scores)
print("Macro AUC: {:.4f}, Micro AUC: {:.4f}".format(auc_macro, auc_micro))

#----------------------------


print("递归调用模型测试")
# 测试模型
y_true = []
y_pred_total = []

model.eval()
x_total = torch.tensor([]).cpu()
z_total = torch.tensor([]).cpu()
quantized_total = torch.tensor([]).cpu()
recon_total = torch.tensor([]).cpu()
y_total = torch.tensor([]).cpu()
y_hat_total = torch.tensor([]).cpu()

start_time = time.time()
#递归调用
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        actual_batch_size = x_batch.size(0)  # 获取当前批次的实际大小
        states = model.init_states(actual_batch_size)  # 使用实际批次大小初始化状态
        # 这里假设x_batch的形状是[batch_size, sequence_length, features]
        h, y_hat, states = model.forward_recurrent(x_batch, states)

        y_pred_t = torch.argmax(y_hat.cpu(), axis=1)

        # 将结果收集到列表中
        y_true += y_batch.tolist()
        y_pred = np.argmax(y_hat.cpu(), axis=1)
        y_pred_total += y_pred.tolist()
        x_total = torch.cat([x_total.cpu(), x_batch.cpu()], dim=0)
        z_total = torch.cat([z_total.cpu(), h.cpu()], dim=0)
        quantized_total = torch.cat([quantized_total.cpu(), y_hat.cpu()], dim=0)
        y_total = torch.cat([y_total.cpu(), y_batch.cpu()], dim=0)
        y_hat_total = torch.cat([y_hat_total.cpu(), y_pred.cpu()], dim=0)

end_time = time.time()
print("Cost time:", end_time - start_time)

acc = accuracy_score(y_true, y_pred_total)
f1_weighted = f1_score(y_true, y_pred_total, average='weighted')
f1_macro = f1_score(y_true, y_pred_total, average='macro')
f1_micro = f1_score(y_true, y_pred_total, average='micro')
precision_macro = precision_score(y_true, y_pred_total, average='macro')
recall_macro = recall_score(y_true, y_pred_total, average='macro')
precision_micro = precision_score(y_true, y_pred_total, average='micro')
recall_micro = recall_score(y_true, y_pred_total, average='micro')

# 打印结果
print('Test Accuracy: {:.4f}'.format(acc))
print('Test Weighted F1 Score: {:.4f}'.format(f1_weighted))
print('Macro F1 Score: {:.4f}'.format(f1_macro))
print('Micro F1 Score: {:.4f}'.format(f1_micro))
print('Macro Precision: {:.4f}, Macro Recall: {:.4f}'.format(precision_macro, recall_macro))
print('Micro Precision: {:.4f}, Micro Recall: {:.4f}'.format(precision_micro, recall_micro))

# 绘制Loss图
# 假设 losses 是你在训练过程中收集的loss列表
# 示例：losses = [0.7, 0.6, 0.5, 0.4, 0.3]
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 二值化预测和真实标签
# 示例中的classes参数应根据实际类别数量进行调整
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14])
y_pred_bin = label_binarize(y_pred_total, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14])

# 分类报告
print(classification_report(y_true, y_pred_total))

# 计算每个类别的AUC及总的AUC
auc_scores = roc_auc_score(y_true_bin, y_pred_bin, average=None)
auc_macro = roc_auc_score(y_true_bin, y_pred_bin, average='macro')
auc_micro = roc_auc_score(y_true_bin, y_pred_bin, average='micro')

# 打印AUC结果
print("AUC Scores by class:", auc_scores)
print("Macro AUC: {:.4f}, Micro AUC: {:.4f}".format(auc_macro, auc_micro))