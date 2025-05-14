#!/usr/bin/env python
# coding: utf-8

# In[1]:
print("测试内容：LSTM NB15小样本多分类Baseline测试")
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
import model.LSTM as LSTM

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
class_number = 10  #分类类别
epochs = 100
proportion = 0.01*5

# In[3]:


#————————————————————————————数据集加载————————————————————————————————
x_train = pd.read_csv("../data/UNSW-NB15/kaggle_UNSW_NB15_full_training.csv", low_memory=False)
num_cols = x_train.shape[1] #获取列数
#提取出最后一列为y
y_train = x_train.pop(x_train.columns[-1]).values

x_test = pd.read_csv("../data/UNSW-NB15/kaggle_UNSW_NB15_full_edit_testing.csv", low_memory=False)
num_cols = x_test.shape[1] #获取列数
y_test = x_test.pop(x_test.columns[-1]).values

# 将读取的数据转化为np格式方便后续训练
x_train = np.array(x_train, dtype=np.float32) # 将数据转换为float32类型
y_train = np.array(y_train, dtype=np.int64) # 将数据转换为int64类型
x_test = np.array(x_test, dtype=np.float32) # 将数据转换为float32类型
y_test = np.array(y_test, dtype=np.int64) # 将数据转换为int64类型

# 合并训练和测试集
x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

# 随机划分训练集，取50%作为训练集
x_train, _, y_train, _ = train_test_split(x_train, y_train, train_size=proportion, random_state=42)

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
model = LSTM.LSTMClassifier(input_size=x_train.shape[1], hidden_size=hidden_dim, output_size=class_number).to(dev)
optimizer = optim.Adam(model.parameters(), lr=learn_rate)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)


# In[5]:


#训练模型
model.train()
losses = []
for epoch in range(epochs):
    start_time = time.time()
    total_loss = 0
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        h,y_hat = model(x_batch)
        loss = LSTM.lstm_loss(y_batch, y_hat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss/len(train_loader))
    print('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, epochs, losses[-1]))
    end_time = time.time()
    print("Cost time:",end_time - start_time)


# In[6]:


from torch.nn import init, Parameter
import matplotlib.patches as mpatches
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

# # ————————————————可视化————————————————
# # 假设你有10个类别
# num_classes = class_number
# cmap = plt.cm.get_cmap("tab20", num_classes)
#
# # 创建一个颜色映射字典，将每个类别映射到一种颜色
# color_dict = {i: cmap(i) for i in range(num_classes)}
#
# # 使用颜色映射字典为每个点分配颜色
# colors_true = [color_dict[int(y)] for y in y_total.cpu()]
# colors_pred = [color_dict[int(y)] for y in y_hat_total.cpu()]
# # 创建一个patch列表，每个类别一个
# patches = [mpatches.Patch(color=color_dict[i], label=f'Class {i}') for i in range(num_classes)]
#
# # 使用PCA来降低数据的维度到3
# params = 'LSTM'
# # 隐藏数据的可视化
# reducer_2D = umap.UMAP(random_state=42)
# Z_pca_origin = reducer_2D.fit_transform(x_total.cpu())
# Z_pca_h = reducer_2D.fit_transform(z_total.cpu())
# Z_pca_VQ = reducer_2D.fit_transform(quantized_total.cpu())
#
# fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# ax1 = axs[0, 0]
# ax2 = axs[0, 1]
# ax3 = axs[1, 0]
# ax4 = axs[1, 1]
#
# # 原始数据的可视化
# scatter1 = ax1.scatter(Z_pca_origin[:, 0], Z_pca_origin[:, 1], c=colors_true, s=20, alpha=0.6, cmap=plt.cm.tab20)
# legend1 = ax1.legend(handles=patches, loc="upper right", title="Classes")
# ax1.add_artist(legend1)
# ax1.set_title("x")
#
# # 原始数据的可视化
# scatter2 = ax2.scatter(Z_pca_h[:, 0], Z_pca_h[:, 1], c=colors_pred, s=20, alpha=0.6)
# legend2 = ax2.legend(handles=patches, loc="upper right", title="Classes")
# ax2.add_artist(legend2)
# ax2.set_title("z")
#
# # VQ后数据的可视化
# scatter3 = ax3.scatter(Z_pca_VQ[:, 0], Z_pca_VQ[:, 1], c=colors_true, s=20, alpha=0.6)
# legend3 = ax3.legend(handles=patches, loc="upper right", title="Classes")
# ax3.add_artist(legend3)
# ax3.set_title("y")
#
# # 预测数据的可视化
# scatter4 = ax4.scatter(Z_pca_VQ[:, 0], Z_pca_VQ[:, 1], c=colors_pred, s=20, alpha=0.6, cmap=plt.cm.tab20)
# legend4 = ax4.legend(handles=patches, loc="upper right", title="Classes")
# ax4.add_artist(legend4)
# ax4.set_title("y_hat")
#
# fig.suptitle(params)
#
# plt.tight_layout()
# # plt.savefig('../output/' + params + '.png')
# plt.show()
# # ————————————————可视化————————————————

# 精确度、召回率、F1分数、AUC的计算
acc = accuracy_score(y_true, y_pred_total)
f1 = f1_score(y_true, y_pred_total, average='weighted')
precision_macro = precision_score(y_true, y_pred_total, average='macro')
recall_macro = recall_score(y_true, y_pred_total, average='macro')
precision_micro = precision_score(y_true, y_pred_total, average='micro')
recall_micro = recall_score(y_true, y_pred_total, average='micro')

# 打印结果
print('Test Accuracy: {:.4f}'.format(acc))
print('Test F1 Score: {:.4f}'.format(f1))
print('Macro Precision: {:.4f}, Macro Recall: {:.4f}'.format(precision_macro, recall_macro))
print('Micro Precision: {:.4f}, Micro Recall: {:.4f}'.format(precision_micro, recall_micro))

# 绘制Loss图
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 二值化预测和真实标签
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y_pred_bin = label_binarize(y_pred_total, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# 分类报告
print(classification_report(y_true, y_pred_total))

# 计算每个类别的AUC及总的AUC
auc_scores = roc_auc_score(y_true_bin, y_pred_bin, average=None)
auc_macro = roc_auc_score(y_true_bin, y_pred_bin, average='macro')
auc_micro = roc_auc_score(y_true_bin, y_pred_bin, average='micro')

# 打印AUC结果
print("AUC Scores by class:", auc_scores)
print("Macro AUC: {:.4f}, Micro AUC: {:.4f}".format(auc_macro, auc_micro))