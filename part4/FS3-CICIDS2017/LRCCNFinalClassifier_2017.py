import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from LRCCNbaseline_2017 import Evaluation  # 用于后续结果评估
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
        self.contamination = 0.1

    def _get_outlier_detector(self, detector_str: str, **kwargs) -> BaseDetector:
        detectors = {
            "CBLOF": CBLOF,
            "IForest": IForest,
            "OCSVM": OCSVM,
            "ROD": ROD,
            # 可以添加更多 PyOD 模型
        }
        if detector_str not in detectors:
            raise ValueError(f"不支持的异常检测模型: {detector_str}")
        return detectors[detector_str](**kwargs)  # 创建模型实例并传入参数

    def fit(self, train_loader, device='cpu'):
        """
        使用In-Domain训练数据拟合异常检测器（仅用嵌入表示，无需标签）。
        """
        self.encoder.eval()
        embeddings_list = []
        with torch.no_grad():
            for x_batch, _ in train_loader:
                x_batch = x_batch.to(device)
                # 获取投影后的嵌入
                _, embeddings = self.encoder(x_batch, return_q=True)
                embeddings = embeddings.view(embeddings.size(0), -1).cpu().numpy()
                embeddings_list.append(embeddings)
        embeddings_all = np.concatenate(embeddings_list, axis=0)
        # 拟合异常检测器，此时内部会基于contamination自动计算阈值
        self.outlier_detector.fit(embeddings_all)
        # 如果需要，也可以保存内置阈值
        self.threshold = self.outlier_detector.threshold_

    def forward(self, x, device='cpu'):
        """
        对输入样本进行OOD检测，并使用encoder的分类器对In-Domain样本做预测。
        对于OOD样本返回-1，对In-Domain样本返回分类预测结果。
        """
        self.encoder.eval()
        with torch.no_grad():
            x = x.to(device)
            logits, embeddings = self.encoder(x, return_q=True)
            embeddings = embeddings.cpu().numpy()

        # 利用PyOD自带的predict方法获取二分类标签（0：In-Domain，1：OOD）
        pred_labels = self.outlier_detector.predict(embeddings)
        logits = logits.cpu().numpy()
        cls_preds = np.argmax(logits, axis=1)
        # 将OOD样本的预测结果标记为-1
        cls_preds = np.where(pred_labels == 1, -1, cls_preds)

        return pred_labels.tolist(), cls_preds.tolist()

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
        detector_params.setdefault('nu', 0.1)  # OCSVM 默认参数
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
            flags, preds = ood_classifier(x_batch, device=device)
            y_pred_total += preds
            ood_detected += flags

    # 输出结果
    print(f"{outlier_detector.__class__.__name__}分类结果：")
    Evaluation(y_true, y_pred_total, title=f"{outlier_detector.__class__.__name__} Classifier")

    # 保存模型
    ood_classifier.save(save_dir='result')
