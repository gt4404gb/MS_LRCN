#!/usr/bin/env python
# coding: utf-8

# In[1]:
print("测试内容：BONUS CIC-IDS2017小样本多分类测试 (适配版)")
# Necessary imports
import math
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             roc_auc_score, roc_curve, auc)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, MinMaxScaler, StandardScaler # Keep relevant ones
from torch.utils.data import DataLoader, TensorDataset

# --- Definitions for BONUS model and Loss (based on previous response) ---

# --- Helper Modules ---
class CausalConv1d(nn.Module):
    """ Implements a 1D causal convolution layer. """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        return x

class MSBlock(nn.Module):
    """ Multi-scale Convolutional Block using Conv1d """
    def __init__(self, in_channels, out_channels_list):
        super(MSBlock, self).__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels_list[0], kernel_size=1, padding='same')
        self.branch2_1 = nn.Conv1d(in_channels, out_channels_list[1], kernel_size=1, padding='same')
        self.branch2_2 = nn.Conv1d(out_channels_list[1], out_channels_list[1], kernel_size=3, padding='same')
        self.branch3_1 = nn.Conv1d(in_channels, out_channels_list[2], kernel_size=1, padding='same')
        self.branch3_2 = nn.Conv1d(out_channels_list[2], out_channels_list[2], kernel_size=3, padding='same')
        self.branch3_3 = nn.Conv1d(out_channels_list[2], out_channels_list[2], kernel_size=3, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.branch1(x))
        out2 = self.relu(self.branch2_1(x)); out2 = self.relu(self.branch2_2(out2))
        out3 = self.relu(self.branch3_1(x)); out3 = self.relu(self.branch3_2(out3)); out3 = self.relu(self.branch3_3(out3))
        out = torch.cat([out1, out2, out3], dim=1)
        return out

class ChannelAttention(nn.Module):
    """ Channel Attention based on Eq 1, 2 """
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, L = x.size()
        query = x.view(batch_size, C, -1)
        key = x.view(batch_size, C, -1).permute(0, 2, 1)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = x.view(batch_size, C, -1)
        out = torch.bmm(attention, value).view(batch_size, C, L)
        out = self.gamma * out + x
        return out

class SpatialAttention1D(nn.Module):
    """ Simplified 1D Spatial Attention """
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(pooled))
        return x * attention

class MRBlock(nn.Module):
    """ Modified Residual Block for ITCNet using CausalConv1d """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(MRBlock, self).__init__()
        self.causal_conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.causal_conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu_out = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.dropout1(self.relu1(self.bn1(self.causal_conv1(x))))
        out = self.dropout2(self.relu2(self.bn2(self.causal_conv2(out))))
        if self.downsample is not None: residual = self.downsample(x)
        out += residual
        return self.relu_out(out)

# --- Main Components ---
class AttMSCNN(nn.Module):
    """ Attention-Enhanced Multi-Scale CNN """
    def __init__(self, input_channels=1):
        super(AttMSCNN, self).__init__()
        # Define based on paper's description (adapted channels for 1D)
        self.msblock1_out_channels = [16, 16, 16]
        self.msblock1 = MSBlock(input_channels, self.msblock1_out_channels)
        self.att1_in_channels = sum(self.msblock1_out_channels)
        self.channel_att1 = ChannelAttention(self.att1_in_channels)
        self.spatial_att1 = SpatialAttention1D()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.msblock2_out_channels = [32, 32, 32]
        self.msblock2 = MSBlock(self.att1_in_channels, self.msblock2_out_channels)
        self.att2_in_channels = sum(self.msblock2_out_channels)
        self.channel_att2 = ChannelAttention(self.att2_in_channels)
        self.spatial_att2 = SpatialAttention1D()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.msblock3_out_channels = [64, 64, 64]
        self.msblock3 = MSBlock(self.att2_in_channels, self.msblock3_out_channels)
        self.att3_in_channels = sum(self.msblock3_out_channels)
        self.channel_att3 = ChannelAttention(self.att3_in_channels)
        self.spatial_att3 = SpatialAttention1D()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.final_out_channels = self.att3_in_channels

    def forward(self, x):
        x = self.pool1(self.spatial_att1(self.channel_att1(self.msblock1(x))))
        x = self.pool2(self.spatial_att2(self.channel_att2(self.msblock2(x))))
        x = self.pool3(self.spatial_att3(self.channel_att3(self.msblock3(x))))
        return x

class ITCNet(nn.Module):
    """ Improved Multi-Scale Temporal Convolutional Network """
    def __init__(self, input_channels, num_channels, kernel_size=2, dropout=0.2, dilations=[1, 2, 4]):
        super(ITCNet, self).__init__()
        self.dilations = dilations
        num_levels = len(dilations)
        self.initial_conv = nn.Conv1d(input_channels, num_channels, 1)
        self.forward_blocks = nn.ModuleList()
        self.backward_blocks = nn.ModuleList()
        for i in range(num_levels):
            dilation_size = dilations[i]
            self.forward_blocks.append(MRBlock(num_channels, num_channels, kernel_size, dilation=dilation_size, dropout=dropout))
            self.backward_blocks.append(MRBlock(num_channels, num_channels, kernel_size, dilation=dilation_size, dropout=dropout))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fusion_weights = nn.Parameter(torch.randn(num_levels))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.initial_conv(x)
        forward_outputs = []
        current_forward = x
        for block in self.forward_blocks:
            current_forward = block(current_forward)
            forward_outputs.append(current_forward)

        x_reversed = torch.flip(x, dims=[2])
        backward_outputs_reversed = []
        current_backward = x_reversed
        for block in self.backward_blocks:
             current_backward = block(current_backward)
             backward_outputs_reversed.append(current_backward)

        fused_outputs_at_scales = []
        for i in range(len(self.dilations)):
            f_out = forward_outputs[i]
            b_out = torch.flip(backward_outputs_reversed[i], dims=[2])
            combined = f_out + b_out
            pooled = self.gap(combined).squeeze(-1)
            fused_outputs_at_scales.append(pooled)

        normalized_weights = self.softmax(self.fusion_weights)
        stacked_outputs = torch.stack(fused_outputs_at_scales, dim=0)
        weighted_outputs = normalized_weights.unsqueeze(-1).unsqueeze(-1) * stacked_outputs
        final_fused_output = torch.sum(weighted_outputs, dim=0)
        return final_fused_output

class Expert(nn.Module):
    """ Simple DNN Expert Network """
    def __init__(self, input_size, hidden_size, output_size):
        super(Expert, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, output_size), nn.ReLU())
    def forward(self, x): return self.net(x)

class Gate(nn.Module):
    """ Gating Network """
    def __init__(self, input_size, num_experts):
        super(Gate, self).__init__()
        self.linear = nn.Linear(input_size, num_experts)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x): return self.softmax(self.linear(x))

# --- BONUS Model ---
class BONUS(nn.Module):
    def __init__(self, input_features, num_classes, itc_channels,
                 num_experts=8, expert_hidden_size=64, expert_out_size=64):
        super(BONUS, self).__init__()
        self.input_features = input_features
        self.num_classes = num_classes
        self.att_mscnn = AttMSCNN(input_channels=1)
        self.att_mscnn_final_channels = self.att_mscnn.final_out_channels
        self.itcnet = ITCNet(input_channels=self.att_mscnn_final_channels, num_channels=itc_channels)
        self.itcnet_out_features = itc_channels
        self.num_experts = num_experts
        gate_input_size = self.itcnet_out_features

        self.experts = nn.ModuleList([Expert(self.itcnet_out_features, expert_hidden_size, expert_out_size) for _ in range(num_experts)])
        self.gate_binary = Gate(gate_input_size, num_experts)
        self.gate_multiclass = Gate(gate_input_size, num_experts)
        self.predictor_binary = nn.Linear(expert_out_size, 1)
        self.predictor_multiclass = nn.Linear(expert_out_size, num_classes)

    def forward(self, x):
        if x.dim() == 2: x_seq = x.unsqueeze(1)
        else: x_seq = x
        att_features = self.att_mscnn(x_seq)
        itc_fused_features = self.itcnet(att_features)
        expert_outputs = [expert(itc_fused_features) for expert in self.experts]
        stacked_expert_outputs = torch.stack(expert_outputs, dim=0)

        gate_b_weights = self.gate_binary(itc_fused_features).permute(1, 0).unsqueeze(-1)
        binary_expert_combination = torch.sum(gate_b_weights * stacked_expert_outputs, dim=0)
        binary_logits = self.predictor_binary(binary_expert_combination).squeeze(-1) # Squeeze last dim

        gate_m_weights = self.gate_multiclass(itc_fused_features).permute(1, 0).unsqueeze(-1)
        multiclass_expert_combination = torch.sum(gate_m_weights * stacked_expert_outputs, dim=0)
        multiclass_logits = self.predictor_multiclass(multiclass_expert_combination)

        return binary_logits, multiclass_logits

class BONUSLoss(nn.Module):
    """ Calculates the combined loss for the BONUS model. """
    def __init__(self, alpha=0.5):
        super(BONUSLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, binary_logits, multiclass_logits, binary_target, multiclass_target):
        loss_b = self.bce_loss(binary_logits, binary_target.float())
        loss_m = self.ce_loss(multiclass_logits, multiclass_target.long())
        total_loss = self.alpha * loss_b + (1.0 - self.alpha) * loss_m
        return total_loss, loss_b, loss_m

# --- End of BONUS definitions ---


#——————————配置调用设备————————————
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device",dev)
#——————————配置调用设备————————————

# Seed for reproducibility
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

# --- Hyperparameters ---
# Data and Training
# input_features will be set after preprocessing
learn_rate = 0.001
batch_size = 1024
epochs = 100
proportion = 0.01
class_number = 15

# BONUS Model Hyperparameters (Examples, adjust based on paper/tuning)
itc_channels = 64          # Channels within ITCNet blocks
num_experts = 16            # Number of experts in MMOE part
expert_hidden_size = 64    # Hidden size within each expert DNN
expert_out_size = 64       # Output size of each expert DNN (input to final predictors)
loss_alpha = 0.5           # Weight for the binary loss component in BONUSLoss

# Class number will be determined dynamically after loading data


# In[3]:

#————————————————————————————数据集加载————————————————————————————————
x = pd.read_csv("../../data/CICIDS2017/cicids2017.csv", low_memory=False)
num_cols = x.shape[1] #获取列数
#提取出最后一列为y
y = x.pop(x.columns[-1]).values

# 将读取的数据转化为np格式方便后续训练
x_train = np.array(y, dtype=np.float32) # 将数据转换为float32类型
y_train = np.array(y, dtype=np.int64) # 将数据转换为int64类型

# 随机划分，比例为80%的训练数据和20%的测试数据
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

# Instantiate the BONUS model
model = BONUS(input_features=204,
              num_classes=class_number, # Use dynamically determined class number
              itc_channels=itc_channels,
              num_experts=num_experts,
              expert_hidden_size=expert_hidden_size,
              expert_out_size=expert_out_size).to(dev)

# Instantiate the BONUS Loss calculator
criterion = BONUSLoss(alpha=loss_alpha).to(dev) # Ensure loss is on same device

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learn_rate)

# Print model parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of parameters: {num_params}')


# In[5]:

# Training loop adapted for BONUS
model.train()

train_losses = []
train_bce_losses = []
train_ce_losses = []

print("\n--- Starting Training ---")
for epoch in range(epochs):
    start_time = time.time()
    epoch_total_loss = 0
    epoch_bce_loss = 0
    epoch_ce_loss = 0
    batch_count = 0

    for x_batch, y_batch_multiclass in train_loader: # y_batch is multi-class target
        # Create binary target: 0 if Normal (assuming class 0), 1 otherwise
        # Adjust this logic if 'Normal' is not class 0
        y_batch_binary = (y_batch_multiclass != 6).long().to(dev) # Convert boolean to long

        optimizer.zero_grad()

        # Forward pass -> get binary and multi-class logits
        binary_logits, multiclass_logits = model(x_batch)

        # Calculate combined loss using BONUSLoss
        total_loss, loss_b, loss_m = criterion(
            binary_logits, multiclass_logits, y_batch_binary, y_batch_multiclass
        )

        # Backward pass
        total_loss.backward()

        # Optimizer step
        optimizer.step()

        epoch_total_loss += total_loss.item()
        epoch_bce_loss += loss_b.item()
        epoch_ce_loss += loss_m.item()
        batch_count += 1

    avg_total_loss = epoch_total_loss / batch_count
    avg_bce_loss = epoch_bce_loss / batch_count
    avg_ce_loss = epoch_ce_loss / batch_count

    train_losses.append(avg_total_loss)
    train_bce_losses.append(avg_bce_loss)
    train_ce_losses.append(avg_ce_loss)

    end_time = time.time()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_total_loss:.6f} '
          f'(BCE: {avg_bce_loss:.6f}, CE: {avg_ce_loss:.6f}), ' # Updated loss names
          f'Time: {end_time - start_time:.2f}s')

print("--- Training Finished ---")


# In[6]:

# Testing loop adapted for BONUS (Evaluating Multi-Class Performance)
print("\n--- Starting Evaluation (Multi-Class Task) ---")
y_true_multiclass = []
y_pred_multiclass = []
y_pred_proba_multiclass = [] # Store probabilities for AUC

model.eval() # Set model to evaluation mode
with torch.no_grad(): # Disable gradient calculations
    for x_batch, y_batch_multiclass in test_loader:

        # Forward pass -> get logits
        binary_logits, multiclass_logits = model(x_batch)

        # Store true multi-class labels
        y_true_multiclass.extend(y_batch_multiclass.tolist())

        # Get predicted multi-class labels from logits
        y_pred = torch.argmax(multiclass_logits, dim=1)
        y_pred_multiclass.extend(y_pred.tolist())

        # Store predicted probabilities for multi-class task for AUC
        y_pred_proba = F.softmax(multiclass_logits, dim=1).cpu().numpy()
        y_pred_proba_multiclass.extend(y_pred_proba)


# Convert list of probabilities to numpy array
y_pred_proba_multiclass = np.array(y_pred_proba_multiclass)

# Calculate metrics for Multi-Class Task
acc = accuracy_score(y_true_multiclass, y_pred_multiclass)
# Use zero_division=0 or 1 to handle classes with no predicted samples
f1 = f1_score(y_true_multiclass, y_pred_multiclass, average='weighted', zero_division=0)

print(f'\nTest Accuracy (Multi-Class): {acc:.4f}')
print(f'Test F1 Score (Multi-Class, Weighted): {f1:.4f}')

# Classification Report for Multi-Class Task
print("\nClassification Report (Multi-Class):")
# Get unique labels present in either true or predicted lists for target_names
labels_present = sorted(list(set(y_true_multiclass) | set(y_pred_multiclass)))
target_names = [f'Class {i}' for i in labels_present]
print(classification_report(y_true_multiclass, y_pred_multiclass, labels=labels_present, target_names=target_names, zero_division=0))

# Plot training loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Total Loss')
plt.plot(train_bce_losses, label='Binary Loss (BCE)', linestyle='--')
plt.plot(train_ce_losses, label='Multi-Class Loss (CE)', linestyle=':')
plt.title('Training Loss Components')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)



# 二值化预测和真实标签
# 示例中的classes参数应根据实际类别数量进行调整
y_true_bin = label_binarize(y_true_multiclass, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14])
y_pred_bin = label_binarize(y_pred_multiclass, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14])

# 分类报告
print(classification_report(y_true_multiclass, y_pred_multiclass))

# 计算每个类别的AUC及总的AUC
auc_scores = roc_auc_score(y_true_bin, y_pred_bin, average=None)
auc_macro = roc_auc_score(y_true_bin, y_pred_bin, average='macro')
auc_micro = roc_auc_score(y_true_bin, y_pred_bin, average='micro')

# 打印AUC结果
print("AUC Scores by class:", auc_scores)
print("Macro AUC: {:.4f}, Micro AUC: {:.4f}".format(auc_macro, auc_micro))