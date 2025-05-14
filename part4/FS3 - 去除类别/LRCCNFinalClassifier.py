import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor, NearestNeighbors
from LRCCNbaseline import Evaluation  # 用于后续结果评估
from pyod.models.lof import LOF
from pyod.models.iforest import  IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.rod import ROD
from pyod.models.cblof import CBLOF
from pyod.models.sos import SOS
from pyod.models.pca import PCA
from pyod.models.base import BaseDetector
import os
import time
import pickle


class LOFOODClassifier(nn.Module):
    """
    在对比学习训练完之后，用 LOF 来做 OOD 检测，
    同时用 KNN 做 In-Domain 分类。

    可配置参数通过构造函数传入，默认值如下：
      - k: KNN 的近邻数 (默认 3)
      - knn_p: KNN 的距离度量参数 (默认 2，即欧氏距离)
      - n_neighbors: LOF 的近邻数 (默认 20)
      - contamination: LOF 中的异常点比例 (默认 0.1)
    """

    def __init__(self, encoder, k=3, knn_p=2, n_neighbors=20, contamination=0.1,threshold=-0.5):
        """
        :param encoder: 已训练好的嵌入网络 (例如对比学习后的 encoder)
        :param k: 用于 KNN 的近邻数，默认 3
        :param knn_p: KNN 距离度量参数，默认 2（欧氏距离）
        :param n_neighbors: 用于 LOF 的近邻数，默认 20
        :param contamination: LOF 中异常点比例，默认 0.1
        """
        super(LOFOODClassifier, self).__init__()
        self.encoder = encoder
        self.k = k
        self.knn = KNeighborsClassifier(n_neighbors=self.k, p=knn_p)
        # novelty=True 表示可以在 fit 后对新数据做 outlier 预测
        self.lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)

        self.threshold = threshold

    def fit_lof_and_knn(self, train_loader, device='cpu'):
        """
        在 In-Domain 的训练集上拟合 LOF（仅用嵌入，不考虑标签）
        以及 KNN（需要标签）。
        """
        self.encoder.eval()
        embeddings_list = []
        labels_list = []

        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                # 假设 encoder 返回 (logits, embeddings)
                logits, embeddings = self.encoder(x_batch, return_q=True)
                # 若需要可将嵌入 reshape 成二维向量
                embeddings = embeddings.view(embeddings.size(0), -1).cpu().numpy()
                embeddings_list.append(embeddings)
                labels_list.append(y_batch.cpu().numpy())

        embeddings_all = np.concatenate(embeddings_list, axis=0)
        labels_all = np.concatenate(labels_list, axis=0)

        # 1) 拟合 LOF：LOF 只关注特征分布，不需要标签
        self.lof.fit(embeddings_all)

        # 2) 训练 KNN 分类器
        self.knn.fit(embeddings_all, labels_all)

    def forward_knn(self, x, device='cpu', threshold=1.0):
        """
        备用 forward 方法：
        根据 LOF 的 decision_function 判断 OOD 样本，
        用 KNN 对 In-Domain 样本进行分类。

        :param x: 输入样本 (Tensor)
        :param device: 运行设备
        :param threshold: LOF decision_function 阈值 (默认 1.0)
        :return: (ood_flags, cls_preds)
            - ood_flags: list，True/False 表示该样本是否被判为 OOD
            - cls_preds: list，对于 In-Domain 样本返回 KNN 分类预测，对于 OOD 返回 -1
        """
        self.encoder.eval()
        with torch.no_grad():
            x = x.to(device)
            logits, embeddings = self.encoder(x, return_q=True)
            embeddings = embeddings.view(embeddings.size(0), -1).cpu().numpy()

        # 使用 LOF 得分，分数越小代表越异常
        lof_scores = self.lof.decision_function(embeddings)
        ood_flags = lof_scores < threshold
        knn_preds = self.knn.predict(embeddings)
        cls_preds = np.where(ood_flags, -1, knn_preds)

        return ood_flags.tolist(), cls_preds.tolist()

    def fit_lof(self, train_loader, device='cpu'):
        """
        在 In-Domain 的训练集上仅拟合 LOF（不训练 KNN）。
        这里使用 encoder 的投影输出（通过 return_q=True）。
        """
        self.encoder.eval()
        embeddings_list = []
        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                # 使用 return_q=True 获取投影后的嵌入
                logits, embeddings = self.encoder(x_batch, return_q=True)
                embeddings = embeddings.view(embeddings.size(0), -1).cpu().numpy()
                embeddings_list.append(embeddings)
        embeddings_all = np.concatenate(embeddings_list, axis=0)
        self.lof.fit(embeddings_all)
        #lof_scores = self.lof.decision_function(embeddings_all)
        #self.threshold = np.percentile(lof_scores, 100 * (1 - self.contamination))


    def forward(self, x, device='cpu', contamination=0.05):
        """
        根据 LOF 判断 OOD 并利用 encoder 的分类头对 In-Domain 样本做预测。

        :param x: 输入样本 (Tensor)
        :param device: 运行设备
        :param threshold: LOF decision_function 阈值 (默认 1.0)
        :return: (ood_flags, cls_preds)
            - ood_flags: list，表示每个样本是否为 OOD (True/False)
            - cls_preds: list，对于 OOD 样本返回 -1，对于 In-Domain 样本返回分类预测
        """
        self.encoder.eval()
        with torch.no_grad():
            x = x.to(device)
            # 使用 return_q=True 以获得投影后的嵌入
            logits, embeddings = self.encoder(x, return_q=True)
            embeddings = embeddings.cpu().numpy()
        '''
        lof_scores = self.lof.decision_function(embeddings)
        #ood_flags = lof_scores < self.threshold
        ood_labels = self.lof.predict(embeddings)
        ood_flags = (ood_labels == 1)

        logits = logits.cpu().numpy()
        cls_preds = np.argmax(logits, axis=1)
        cls_preds = np.where(ood_flags, -1, cls_preds)
        '''

        # 使用 LOF 判断 OOD
        lof_scores = self.lof.decision_function(embeddings)  # 分数越小 => 越异常
        ood_flags = lof_scores < self.threshold  # 一次性判断所有样本是否 OOD

        # 对于 In-Domain 样本，使用分类头预测类别
        logits = logits.cpu().numpy()
        cls_preds = np.argmax(logits, axis=1)  # 分类头返回类别索引

        # 对 OOD 样本赋值 -1
        cls_preds = np.where(ood_flags, -1, cls_preds)  # 如果是 OOD，则赋值为 -1
        ood_flags = ood_flags.tolist()  # 转为 list
        cls_preds = cls_preds.tolist()  # 转为 list


        return ood_flags, cls_preds

    def save(self, save_dir='result'):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        encoder_path = os.path.join(save_dir, f'encoder_LOF_{timestamp}.pth')
        lof_path = os.path.join(save_dir, f'lof_{timestamp}.pkl')

        torch.save(self.encoder.state_dict(), encoder_path)
        print(f"Encoder saved to {encoder_path}")
        with open(lof_path, 'wb') as f:
            pickle.dump(self.lof, f)
        print(f"LOF saved to {lof_path}")

class ODINClassifier(nn.Module):
    """
    使用 ODIN 方法进行 OOD 检测与分类。

    :param encoder: 已训练好的编码器，要求在前向传播时返回 (logits, embeddings)
    :param temperature: 温度缩放参数 T，默认 1000
    :param epsilon: 输入扰动幅度，默认 0.0014
    :param threshold: OOD 检测阈值，默认 0.5
    """

    def __init__(self, encoder):
        super(ODINClassifier, self).__init__()
        self.encoder = encoder

    def forward(self, x, return_q=False):
        """
        对输入样本 x 应用 ODIN 方法进行 OOD 检测和分类。

        :param x: 输入样本 (Tensor)
        :param device: 运行设备
        :return: (ood_flags, cls_preds)
            - ood_flags: list，表示每个样本是否为 OOD (True/False)
            - cls_preds: list，对于 OOD 样本返回 -1，对于 In-Domain 样本返回分类预测
        """
        with torch.no_grad():
            # 使用 return_q=True 以获得投影后的嵌入
            logits, embeddings = self.encoder(x, return_q=return_q)
        return logits, embeddings

    def save(self, save_dir='result'):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        encoder_path = os.path.join(save_dir, f'encoder_ODIN_{timestamp}.pth')

        torch.save(self.encoder.state_dict(), encoder_path)
        print(f"Encoder saved to {encoder_path}")

class ConfigurableOODClassifier(nn.Module):
    """
    可配置的OOD检测分类器，使用指定的异常检测模型进行OOD检测，
    并使用encoder的分类器对In-Domain样本进行分类。

    参数：
        encoder: 已训练好的嵌入网络（例如对比学习后的encoder）
        outlier_detector: PyOD的异常检测模型实例（如LOF、Isolation Forest等）
    """

    def __init__(self, encoder, outlier_detector_str: str, **kwargs):
        super(ConfigurableOODClassifier, self).__init__()
        self.encoder = encoder
        self.outlier_detector = self._get_outlier_detector(outlier_detector_str, **kwargs)
        self.threshold = 0
        self.contamination = 0.05

    def _get_outlier_detector(self, detector_str: str, **kwargs) -> BaseDetector:
        detectors = {
            "CBLOF": CBLOF,
            "IForest": IForest,
            "OCSVM": OCSVM,
            "ROD": ROD,
            "SOS": SOS,
            # 可以添加更多 PyOD 模型
        }
        if detector_str not in detectors:
            raise ValueError(f"不支持的异常检测模型: {detector_str}")
        return detectors[detector_str](**kwargs)  # 创建模型实例并传入参数

    def fit(self, train_loader, device='cpu'):
        """
        在In-Domain的训练集上拟合异常检测模型（仅使用嵌入，不考虑标签）。

        参数：
            train_loader: 训练数据加载器
            device: 计算设备（默认 'cpu'）
        """
        self.encoder.eval()
        embeddings_list = []
        with torch.no_grad():
            for x_batch, _ in train_loader:
                x_batch = x_batch.to(device)
                # 使用 return_q=True 获取投影后的嵌入
                _, embeddings = self.encoder(x_batch, return_q=True)
                embeddings = embeddings.view(embeddings.size(0), -1).cpu().numpy()
                embeddings_list.append(embeddings)
        embeddings_all = np.concatenate(embeddings_list, axis=0)
        self.outlier_detector.fit(embeddings_all)
        scores = self.outlier_detector.decision_function(embeddings_all)
        self.threshold = np.percentile(scores, 100 * (1 - self.contamination))

    def forward(self, x, device='cpu', contamination=0.05):
        """
        根据异常检测模型判断OOD并利用encoder的分类器对In-Domain样本做预测。

        参数：
            x: 输入样本（Tensor）
            device: 运行设备（默认 'cpu'）
            threshold: 异常检测模型的decision_function阈值（默认 0.0）

        返回：
            ood_flags: list，表示每个样本是否为OOD (True/False)
            cls_preds: list，对于OOD样本返回-1，对于In-Domain样本返回分类预测
        """
        self.encoder.eval()
        with torch.no_grad():
            x = x.to(device)
            # 使用 return_q=True 以获得投影后的嵌入
            logits, embeddings = self.encoder(x, return_q=True)
            embeddings = embeddings.cpu().numpy()

        # 使用异常检测模型的decision_function获取异常分数
        scores = self.outlier_detector.decision_function(embeddings)
        # 注意：不同模型的异常分数定义不同
        # - LOF: scores越小越异常（负值）
        # - Isolation Forest等: scores越大越异常
        # 这里假设使用LOF-like模型，scores < threshold 为OOD
        ood_flags = scores > self.threshold

        logits = logits.cpu().numpy()
        cls_preds = np.argmax(logits, axis=1)
        cls_preds = np.where(ood_flags, -1, cls_preds)

        return ood_flags.tolist(), cls_preds.tolist()

    def save(self, save_dir='result'):
        """
        保存模型到指定文件夹。

        参数：
            save_dir: 保存路径（默认 'result'）
        """
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        # 生成保存文件名（基于模型类型和时间戳）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        detector_name = self.outlier_detector.__class__.__name__
        encoder_path = os.path.join(save_dir, f'encoder_{detector_name}_{timestamp}.pth')
        detector_path = os.path.join(save_dir, f'detector_{detector_name}_{timestamp}.pkl')

        # 保存 encoder (PyTorch 模型)
        torch.save(self.encoder.state_dict(), encoder_path)
        print(f"Encoder saved to {encoder_path}")

        # 保存 outlier_detector (PyOD 模型)
        with open(detector_path, 'wb') as f:
            pickle.dump(self.outlier_detector, f)
        print(f"Outlier detector saved to {detector_path}")

def LOFtest(encoder, train_loader2, test_loader, class_number, device='cpu', lof_params=None, contamination=0.05):
    """
    使用 LOF+KNN 进行 OOD 检测与分类演示。

    :param encoder: 已训练好的 encoder
    :param train_loader2: 用于拟合 LOF/KNN 的训练数据加载器（视为 In-Domain）
    :param test_loader: 测试数据加载器
    :param class_number: 类别数（接口保留，可根据需要扩展）
    :param device: 计算设备
    :param lof_params: dict，包含 LOFOODClassifier 的配置参数，例如：
           {'k': 5, 'knn_p': 2, 'n_neighbors': 15, 'contamination': 0.02}
           若为 None，则使用默认参数
    :param threshold: LOF decision_function 阈值，默认 0.0
    """
    if lof_params is None:
        lof_params = {'k': 5, 'knn_p': 2, 'n_neighbors': 15, 'contamination': 0.02,'threshold':-0.5}
    lof_ood_classifier = LOFOODClassifier(encoder, **lof_params).to(device)

    # 这里调用 fit_lof 方法拟合 LOF（仅使用嵌入）
    lof_ood_classifier.fit_lof(train_loader2, device=device)

    lof_ood_classifier.eval()
    y_true = []
    y_pred_total = []
    ood_detected = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_true += y_batch.tolist()
            flags, preds = lof_ood_classifier(x_batch, device=device, contamination=contamination)
            y_pred_total += preds
            ood_detected += flags

    print("LOF分类结果：")
    f1 =Evaluation(y_true, y_pred_total, title="LOF Classifier"+str(lof_params))
    #lof_ood_classifier.save(save_dir='result')

    return f1

def KNNtest(encoder, train_loader, test_loader, class_number, device='cpu', knn_params=None):
    """
    使用 KNN 进行 In-Domain 分类的对比实验。

    :param encoder: 已训练好的 encoder
    :param train_loader: 用于拟合 KNN 的训练数据加载器（视为 In-Domain 数据）
    :param test_loader: 测试数据加载器
    :param class_number: 类别数（目前不直接用于 KNN 分类，但预留接口）
    :param device: 计算设备
    :param knn_params: dict，包含 KNN 的配置参数，例如：
           {'k': 3, 'knn_p': 2}
           若为 None，则使用默认参数
    """
    if knn_params is None:
        knn_params = {'k': 3, 'knn_p': 2}

    # 实例化 LOFOODClassifier，但这里只关注其中的 KNN 分类器部分
    knn_classifier = LOFOODClassifier(encoder, **knn_params).to(device)

    # 使用训练数据拟合 KNN（fit_lof_and_knn 同时拟合 LOF，但后续测试阶段仅使用 KNN 部分）
    knn_classifier.fit_lof_and_knn(train_loader, device=device)

    knn_classifier.eval()
    y_true = []
    y_pred_total = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_true += y_batch.tolist()
            x_batch = x_batch.to(device)
            # 使用 encoder 获取嵌入（注意这里与 fit_lof_and_knn 中调用的方式保持一致）
            logits, embeddings = encoder(x_batch, return_q=True)
            embeddings = embeddings.view(embeddings.size(0), -1).cpu().numpy()
            # 直接使用 KNN 分类器进行预测（忽略 LOF 部分）
            knn_preds = knn_classifier.knn.predict(embeddings)
            y_pred_total += knn_preds.tolist()

    print("KNN 基线分类结果：")
    Evaluation(y_true, y_pred_total, title="KNN Classifier")

    def LOF_KNN_test(encoder, train_loader, test_loader, class_number, device='cpu',
                     lof_params=None, threshold=1.0):
        """
        使用 LOF + KNN 方法进行 OOD 检测与分类的对比实验。

        :param encoder: 已训练好的 encoder，要求在前向传播时返回 (logits, embeddings)
        :param train_loader: 用于拟合 LOF 和训练 KNN 的训练数据加载器（视为 In-Domain 数据）
        :param test_loader: 测试数据加载器
        :param class_number: 类别数（接口预留，可根据需要扩展）
        :param device: 计算设备
        :param lof_params: dict，包含 LOFOODClassifier 的配置参数，例如：
               {'k': 5, 'knn_p': 2, 'n_neighbors': 15, 'contamination': 0.02}
               若为 None，则使用默认参数
        :param threshold: LOF decision_function 阈值，默认 1.0，
                          小于该阈值的样本视为 OOD，预测类别置为 -1
        """
        if lof_params is None:
            lof_params = {'k': 5, 'knn_p': 2, 'n_neighbors': 15, 'contamination': 0.02}

        # 实例化 LOFOODClassifier，并在训练数据上同时拟合 LOF 和 KNN
        lof_knn_classifier = LOFOODClassifier(encoder, **lof_params).to(device)
        lof_knn_classifier.fit_lof_and_knn(train_loader, device=device)

        lof_knn_classifier.eval()
        y_true = []
        y_pred_total = []
        ood_detected = []

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                y_true += y_batch.tolist()
                # 调用 forward_knn 使用 LOF 判断 OOD，且对 In-Domain 样本使用 KNN 分类器进行预测
                flags, preds = lof_knn_classifier.forward_knn(x_batch, device=device, threshold=threshold)
                y_pred_total += preds
                ood_detected += flags

        print("LOF + KNN 对比实验结果：")
        Evaluation(y_true, y_pred_total, title="LOF+KNN Classifier")

def LOF_KNN_test(encoder, train_loader, test_loader, class_number, device='cpu',lof_params=None, threshold=1.0):
    """
    使用 LOF + KNN 方法进行 OOD 检测与分类的对比实验。

    :param encoder: 已训练好的 encoder，要求在前向传播时返回 (logits, embeddings)
    :param train_loader: 用于拟合 LOF 和训练 KNN 的训练数据加载器（视为 In-Domain 数据）
    :param test_loader: 测试数据加载器
    :param class_number: 类别数（接口预留，可根据需要扩展）
    :param device: 计算设备
    :param lof_params: dict，包含 LOFOODClassifier 的配置参数，例如：
           {'k': 5, 'knn_p': 2, 'n_neighbors': 15, 'contamination': 0.02}
           若为 None，则使用默认参数
    :param threshold: LOF decision_function 阈值，默认 1.0，
                      小于该阈值的样本视为 OOD，预测类别置为 -1
    """
    if lof_params is None:
        lof_params = {'k': 5, 'knn_p': 2, 'n_neighbors': 15, 'contamination': 0.02}

    # 实例化 LOFOODClassifier，并在训练数据上同时拟合 LOF 和 KNN
    lof_knn_classifier = LOFOODClassifier(encoder, **lof_params).to(device)
    lof_knn_classifier.fit_lof_and_knn(train_loader, device=device)

    lof_knn_classifier.eval()
    y_true = []
    y_pred_total = []
    ood_detected = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_true += y_batch.tolist()
            # 调用 forward_bak 使用 LOF 判断 OOD，且对 In-Domain 样本使用 KNN 分类器进行预测
            flags, preds = lof_knn_classifier.forward_knn(x_batch, device=device, threshold=threshold)
            y_pred_total += preds
            ood_detected += flags

    print("LOF + KNN 对比实验结果：")
    Evaluation(y_true, y_pred_total, title="LOF+KNN Classifier")

def ODINtest(encoder, test_loader, device='cpu', temperature=1, epsilon=0.0014, threshold=0.5):
    """
    使用 ODIN 方法进行 OOD 检测与分类的对比实验。

    :param encoder: 已训练好的模型或 encoder，要求输出 logits（若 encoder 返回 tuple，则取第一个元素）
    :param test_loader: 测试数据加载器
    :param device: 计算设备
    :param temperature: 温度缩放参数，默认设置为 1000（可根据实际情况调整）
    :param epsilon: 输入扰动幅度，默认 0.0014（可根据实际情况调整）
    :param threshold: ODIN 的决策阈值，默认 0.5，若最大 softmax 概率低于该阈值，则认为样本为 OOD
    """
    encoder.eval()
    y_true = []
    y_pred_total = []
    ood_detected = []
    # 1. 创建一个列表来存储每个 batch 的 logits
    logits_pert_list = []
    softmax = nn.Softmax(dim=1)

    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        # 开启输入梯度计算，用于生成扰动
        x_batch.requires_grad = True

        # 第一次前向传播，获得原始 logits
        logits = encoder(x_batch)
        # 若 encoder 返回 tuple，则取第一个输出（通常为 logits）
        if isinstance(logits, tuple):
            logits = logits[0]
        logits_temp = logits / temperature
        prob = softmax(logits_temp)
        # 获得预测类别
        preds = torch.argmax(prob, dim=1)

        # 针对每个样本，计算预测类别对应的负对数概率作为 loss
        loss = -torch.log(prob[range(x_batch.size(0)), preds])
        encoder.zero_grad()
        loss.sum().backward()

        # 计算输入梯度的符号，并生成扰动后的输入
        gradient_sign = x_batch.grad.data.sign()
        x_batch_pert = x_batch - epsilon * gradient_sign
        x_batch_pert = x_batch_pert.detach()

        # 对扰动后的输入进行第二次前向传播
        logits_pert = encoder(x_batch_pert)
        if isinstance(logits_pert, tuple):
            logits_pert = logits_pert[0]

        # 2. 将当前 batch 的 logits 添加到列表 (detach 以免内存泄漏)
        #    如果后续评估在 CPU 上，可以 .cpu() 提前转移，减少 GPU 显存占用
        logits_pert_list.append(logits_pert.detach())

        logits_pert_temp = logits_pert / temperature
        prob_pert = softmax(logits_pert_temp)
        # 计算扰动后每个样本的最大 softmax 概率
        max_prob, _ = torch.max(prob_pert, dim=1)

        # 根据阈值判断 OOD：低于阈值认为是 OOD（用 -1 表示），否则仍使用原预测类别
        ood_flag = max_prob < threshold
        preds_final = preds.clone()
        preds_final[ood_flag] = -1

        y_true.extend(y_batch.cpu().tolist())
        y_pred_total.extend(preds_final.cpu().tolist())
        ood_detected.extend(ood_flag.cpu().tolist())

    # 3. 在循环结束后，进行一次拼接
    logits_pert_total = torch.cat(logits_pert_list, dim=0)

    print("ODIN 对比实验结果：")
    f1_score = Evaluation(y_true, y_pred_total,logits_pert_total.cpu().numpy(), title="ODIN Classifier")
    return  f1_score

def ConfigurableOODtest(encoder, train_loader, test_loader, class_number, device='cpu', outlier_detector_str="LOF", detector_params=None, contamination=0.05):
    """
    使用可配置的OOD检测分类器进行测试，支持多种异常检测方法（如LOF、IForest、ROD、OCSVM）。

    参数：
        encoder: 已训练好的encoder
        train_loader: 用于拟合异常检测模型的训练数据加载器（视为In-Domain）
        test_loader: 测试数据加载器
        class_number: 类别数（接口保留，可根据需要扩展）
        device: 计算设备（默认 'cpu'）
        outlier_detector_str: 字符串，指定异常检测模型，可选值包括：
                             "LOF" (默认), "IForest", "ROD", "OCSVM"
        detector_params: dict，指定异常检测模型的参数，若为None则使用默认参数
        contamination: 异常点比例（默认 0.05）
    """
    # 设置默认参数，若未提供 detector_params
    if detector_params is None:
        detector_params = {}

    # 根据 outlier_detector_str 初始化对应的 PyOD 模型
    if outlier_detector_str == "LOF":
        detector_params.setdefault('n_neighbors', 20)  # LOF 默认参数
        detector_params.setdefault('contamination', contamination)
        outlier_detector = LOF(**detector_params)
    elif outlier_detector_str == "IForest":
        detector_params.setdefault('n_estimators', 100)  # Isolation Forest 默认参数
        detector_params.setdefault('contamination', contamination)
        outlier_detector = IForest(**detector_params)
    elif outlier_detector_str == "ROD":
        detector_params.setdefault('contamination', contamination)
        outlier_detector = ROD(**detector_params)
    elif outlier_detector_str == "OCSVM":
        detector_params.setdefault('nu', 0.01)  # OCSVM 默认参数
        detector_params.setdefault('kernel', 'rbf')
        outlier_detector = OCSVM(**detector_params)
    elif outlier_detector_str == "SOS":
        detector_params.setdefault('contamination', contamination)
        outlier_detector = SOS(**detector_params)
    elif outlier_detector_str == "PCA":
        detector_params.setdefault('contamination', contamination)
        outlier_detector = PCA(**detector_params)
    else:
        raise ValueError(f"不支持的异常检测模型: {outlier_detector_str}. "
                         f"支持的模型包括: 'LOF', 'IForest', 'ROD', 'OCSVM', 'SOS', 'PCA'")

    # 实例化 ConfigurableOODClassifier
    ood_classifier = ConfigurableOODClassifier(encoder, outlier_detector_str=outlier_detector_str,
                                               **detector_params).to(device)

    # 拟合异常检测模型
    ood_classifier.fit(train_loader, device=device)

    # 测试阶段
    ood_classifier.eval()
    y_true = []
    y_pred_total = []
    ood_detected = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_true += y_batch.tolist()
            flags, preds = ood_classifier(x_batch, device=device, contamination=contamination)
            y_pred_total += preds
            ood_detected += flags

    # 输出结果
    print(f"{outlier_detector.__class__.__name__}分类结果：")
    Evaluation(y_true, y_pred_total, title=f"{outlier_detector.__class__.__name__} Classifier")

    # 保存模型
    ood_classifier.save(save_dir='result')


class INFLOODClassifier(nn.Module):
    """
    在对比学习训练完之后，用 INFLO（Influence Outlier Detection）来做 OOD 检测，
    同时用 KNN 做 In-Domain 分类。

    可配置参数通过构造函数传入，默认值如下：
      - k: KNN 的近邻数 (默认 3)
      - knn_p: KNN 的距离度量参数 (默认 2，即欧氏距离)
      - n_neighbors: INFLO 的近邻数 (默认 20)
      - contamination: INFLO 中的异常点比例 (默认 0.1)
      - threshold: INFLO 分数的阈值 (默认 -0.5)
    """

    def __init__(self, encoder, k=3, knn_p=2, n_neighbors=20, contamination=0.1, threshold=-0.5):
        """
        :param encoder: 已训练好的嵌入网络 (例如对比学习后的 encoder)
        :param k: 用于 KNN 的近邻数，默认 3
        :param knn_p: KNN 距离度量参数，默认 2（欧氏距离）
        :param n_neighbors: 用于 INFLO 的近邻数，默认 20
        :param contamination: INFLO 中异常点比例，默认 0.1
        :param threshold: INFLO 分数阈值，默认 -0.5
        """
        super(INFLOODClassifier, self).__init__()
        self.encoder = encoder
        self.k = k
        self.knn_p = knn_p
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.threshold = threshold

        # 用于计算近邻和反向近邻
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, p=knn_p)
        self.train_embeddings = None  # 存储训练集嵌入

    def fit_inflo(self, train_loader, device='cpu'):
        """
        在 In-Domain 的训练集上拟合 INFLO。
        这里使用 encoder 的投影输出（通过 return_q=True）。
        """
        self.encoder.eval()
        embeddings_list = []
        with torch.no_grad():
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                logits, embeddings = self.encoder(x_batch, return_q=True)
                embeddings = embeddings.view(embeddings.size(0), -1).cpu().numpy()
                embeddings_list.append(embeddings)
        self.train_embeddings = np.concatenate(embeddings_list, axis=0)

        # 拟合 NearestNeighbors 以计算近邻和反向近邻
        self.nn.fit(self.train_embeddings)

        # 计算 INFLO 分数的核心逻辑
        distances, indices = self.nn.kneighbors(self.train_embeddings)
        inflo_scores = self._compute_inflo_scores(self.train_embeddings, distances, indices)

        # 根据 contamination 设置初始阈值（可选）
        self.threshold = np.percentile(inflo_scores, 100 * (1 - self.contamination))
        print(f"INFLO 初始阈值设置为: {self.threshold}")

    def _compute_inflo_scores(self, embeddings, distances, indices):
        """
        计算 INFLO 分数。
        :param embeddings: 训练集嵌入
        :param distances: k 近邻的距离
        :param indices: k 近邻的索引
        :return: INFLO 分数数组
        """
        n_samples = embeddings.shape[0]
        inflo_scores = np.zeros(n_samples)

        # 计算每个点的局部密度和反向近邻
        for i in range(n_samples):
            # k 近邻集合
            knn_indices = indices[i]
            knn_distances = distances[i]

            # 计算反向近邻 (RNN): 哪些点将 i 视为近邻
            rnn_indices = []
            for j in range(n_samples):
                if i in indices[j]:
                    rnn_indices.append(j)

            # 合并近邻和反向近邻，构成 influence space
            influence_space = list(set(knn_indices) | set(rnn_indices))
            if not influence_space:
                inflo_scores[i] = 0  # 避免空集情况
                continue

            # 计算局部密度 (使用平均距离的倒数)
            local_density = 1.0 / (np.mean(knn_distances) + 1e-10)  # 避免除以零

            # 计算 influence space 中所有点的平均密度
            influence_densities = []
            for idx in influence_space:
                neighbor_distances = self.nn.kneighbors(embeddings[idx].reshape(1, -1), return_distance=True)[0][0]
                influence_densities.append(1.0 / (np.mean(neighbor_distances) + 1e-10))
            avg_influence_density = np.mean(influence_densities)

            # INFLO 分数 = 局部密度 / 平均影响密度
            inflo_scores[i] = local_density / (avg_influence_density + 1e-10) if avg_influence_density > 0 else 0

        # INFLO 分数越小越异常，与 LOF 一致
        return -inflo_scores  # 取负值，与 LOF 的 decision_function 对齐

    def forward(self, x, device='cpu', contamination=0.05):
        """
        根据 INFLO 判断 OOD 并利用 encoder 的分类头对 In-Domain 样本做预测。

        :param x: 输入样本 (Tensor)
        :param device: 运行设备
        :param contamination: 用于动态调整（未使用，仅为兼容性保留）
        :return: (ood_flags, cls_preds)
            - ood_flags: list，表示每个样本是否为 OOD (True/False)
            - cls_preds: list，对于 OOD 样本返回 -1，对于 In-Domain 样本返回分类预测
        """
        self.encoder.eval()
        with torch.no_grad():
            x = x.to(device)
            logits, embeddings = self.encoder(x, return_q=True)
            embeddings = embeddings.cpu().numpy()

        # 计算测试样本的 INFLO 分数
        distances, indices = self.nn.kneighbors(embeddings)
        inflo_scores = self._compute_inflo_scores(embeddings, distances, indices)

        # 使用阈值判断 OOD
        ood_flags = inflo_scores < self.threshold  # 分数小于阈值视为 OOD

        # 对于 In-Domain 样本，使用分类头预测类别
        logits = logits.cpu().numpy()
        cls_preds = np.argmax(logits, axis=1)  # 分类头返回类别索引

        # 对 OOD 样本赋值 -1
        cls_preds = np.where(ood_flags, -1, cls_preds)
        ood_flags = ood_flags.tolist()
        cls_preds = cls_preds.tolist()

        return ood_flags, cls_preds

    def save(self, save_dir='result'):
        """
        保存 encoder 和 INFLO 模型。
        """
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        encoder_path = os.path.join(save_dir, f'encoder_INFLO_{timestamp}.pth')
        nn_path = os.path.join(save_dir, f'nn_INFLO_{timestamp}.pkl')

        torch.save(self.encoder.state_dict(), encoder_path)
        print(f"Encoder saved to {encoder_path}")
        with open(nn_path, 'wb') as f:
            pickle.dump(self.nn, f)
        print(f"NearestNeighbors (INFLO) saved to {nn_path}")


def INFLOtest(encoder, train_loader2, test_loader, class_number, device='cpu', inflo_params=None, contamination=0.05):
    """
    使用 INFLO 进行 OOD 检测与分类演示。

    :param encoder: 已训练好的 encoder
    :param train_loader2: 用于拟合 INFLO 的训练数据加载器（视为 In-Domain）
    :param test_loader: 测试数据加载器
    :param class_number: 类别数（接口保留，可根据需要扩展）
    :param device: 计算设备
    :param inflo_params: dict，包含 INFLOODClassifier 的配置参数，例如：
           {'k': 5, 'knn_p': 2, 'n_neighbors': 15, 'contamination': 0.02, 'threshold': -0.5}
           若为 None，则使用默认参数
    :param contamination: 用于动态调整（仅为兼容性保留，未直接使用）
    :return: F1 分数
    """
    if inflo_params is None:
        inflo_params = {'k': 5, 'knn_p': 2, 'n_neighbors': 15, 'contamination': 0.02, 'threshold': -0.5}

    # 初始化 INFLOODClassifier
    inflo_ood_classifier = INFLOODClassifier(encoder, **inflo_params).to(device)

    # 拟合 INFLO
    inflo_ood_classifier.fit_inflo(train_loader2, device=device)

    inflo_ood_classifier.eval()
    y_true = []
    y_pred_total = []
    ood_detected = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            y_true += y_batch.tolist()
            flags, preds = inflo_ood_classifier(x_batch, device=device, contamination=contamination)
            y_pred_total += preds
            ood_detected += flags

    print("INFLO 分类结果：")
    f1 = Evaluation(y_true, y_pred_total, title="INFLO Classifier" + str(inflo_params))
    # inflo_ood_classifier.save(save_dir='result')  # 可选保存模型，注释与原代码一致

    return f1



class GODINClassifier(nn.Module):
    """
    使用 G-ODIN 方法进行 OOD 检测与分类。

    :param encoder: 已训练好的编码器，要求在前向传播时返回 (logits, embeddings)
    :param threshold: OOD 检测阈值，默认 0.5
    """

    def __init__(self, encoder):
        super(GODINClassifier, self).__init__()
        self.encoder = encoder

    def forward(self, x, return_q=False):
        """
        对输入样本 x 应用 G-ODIN 方法进行 OOD 检测和分类。

        :param x: 输入样本 (Tensor)
        :param return_q: 是否返回投影后的嵌入
        :return: (logits, embeddings)
        """
        with torch.no_grad():
            # 使用 return_q=True 以获得投影后的嵌入
            logits, embeddings = self.encoder(x, return_q=return_q)
        return logits, embeddings

    def save(self, save_dir='result'):
        os.makedirs(save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        encoder_path = os.path.join(save_dir, f'encoder_GODIN_{timestamp}.pth')

        torch.save(self.encoder.state_dict(), encoder_path)
        print(f"Encoder saved to {encoder_path}")


def GODINtest(encoder, test_loader, device='cpu', threshold=0.5):
    """
    使用 G-ODIN 方法进行 OOD 检测与分类的对比实验。

    :param encoder: 已训练好的模型或 encoder，要求输出 (logits, embeddings)
    :param test_loader: 测试数据加载器
    :param device: 计算设备
    :param threshold: G-ODIN 的决策阈值，默认 0.5，若置信度得分低于该阈值，则认为样本为 OOD
    """
    encoder.eval()
    y_true = []
    y_pred_total = []
    ood_detected = []
    softmax = nn.Softmax(dim=1)

    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)

        # 前向传播，获取 logits 和 embeddings
        logits, embeddings = encoder(x_batch, return_q=True)  # 假设 encoder 返回 (logits, embeddings)

        # 计算 softmax 概率
        prob = softmax(logits)
        preds = torch.argmax(prob, dim=1)

        # G-ODIN 的得分函数：结合 softmax 最大概率和嵌入的范数
        max_prob, _ = torch.max(prob, dim=1)  # 最大 softmax 概率
        embed_norm = torch.norm(embeddings, p=2, dim=1)  # 嵌入的 L2 范数
        embeddings = embeddings / embed_norm.unsqueeze(1)  # 归一化嵌入
        embed_norm = torch.norm(embeddings, p=2, dim=1)
        confidence_score = max_prob * torch.exp(-embed_norm)  # G-ODIN 的置信度得分

        # 根据阈值判断 OOD：低于阈值认为是 OOD（用 -1 表示）
        ood_flag = confidence_score < threshold
        preds_final = preds.clone()
        preds_final[ood_flag] = -1

        y_true.extend(y_batch.cpu().tolist())
        y_pred_total.extend(preds_final.cpu().tolist())
        ood_detected.extend(ood_flag.cpu().tolist())

    print("G-ODIN 对比实验结果：")
    Evaluation(y_true, y_pred_total, title="G-ODIN Classifier")  # 假设 Evaluation 是已定义的评估函数