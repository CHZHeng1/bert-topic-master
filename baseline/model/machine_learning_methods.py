import numpy as np
import pandas as pd

import torch
from torch import nn
from topic_model_sklearn.topic_loader import load_document_topics


def data_process_for_ml(data_filepath, document_topics_filepath, manual_features_filepath):
    """
    传统机器学习方法---导入数据和标签
    :param data_filepath: 数据集路径（为了获取标签）
    :param document_topics_filepath: 文档主题分布路径
    :param manual_features_filepath: 微博影响力和用户可信度路径
    """
    raw_iter = pd.read_csv(data_filepath)
    # 标签
    labels = [raw[-1] for raw in raw_iter.values]
    # 文档主题分布
    topic_dists = load_document_topics(document_topics_filepath)
    # 微博影响力和用户可信度
    manual_features = np.load(manual_features_filepath)
    # 特征合并
    features = np.concatenate((topic_dists, manual_features), axis=1)
    return features, labels


def load_data_for_ml(config):
    """加载数据"""
    train_features, train_labels = data_process_for_ml(config.train_data_filepath,
                                                       config.train_document_topics_filepath,
                                                       config.train_manual_features_filepath)
    test_features, test_labels = data_process_for_ml(config.test_data_filepath,
                                                     config.test_document_topics_filepath,
                                                     config.test_manual_features_filepath)
    return train_features, test_features, train_labels, test_labels


class LogisticRegression(nn.Module):
    """逻辑回归"""
    def __init__(self, input_dim=27, num_class=2):
        super(LogisticRegression, self).__init__()
        self.input_layer = nn.Linear(input_dim, input_dim)
        self.normalize = nn.BatchNorm1d(input_dim)
        self.classifier = nn.Linear(input_dim, num_class)
        self.activate = torch.sigmoid

    def forward(self, inputs):
        input_layer_out = self.normalize(self.input_layer(inputs))
        outputs = self.activate(self.classifier(input_layer_out))
        return outputs


# if __name__ == '__main__':
#     ml_config = Config()
#     Xtrain, Xtest, Ytrain, Ytest = load_data_for_ml(ml_config)




