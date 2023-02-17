from datetime import datetime
import pandas as pd
import numpy as np

from config import ModelConfig


def get_base_variable(data_file_path):
    """获取基础特征变量"""
    raw_iter = pd.read_csv(data_file_path)
    base_variable = []
    for ind, raw in enumerate(raw_iter.values):
        end_time = datetime.strptime('2022-09-05 00:00:00', '%Y-%m-%d %H:%M:%S')
        released_time = datetime.strptime(raw[2], '%Y-%m-%d %H:%M:%S')
        duration = end_time - released_time
        time_interval = duration.days  # 按天计算时间差
        like = raw[3]
        repost = raw[4]
        comment = raw[5]
        num = raw[8]
        following = raw[9]
        fans = raw[10]
        if_verified = raw[11]
        variable = [time_interval, like, repost, comment, num, following, fans, if_verified]
        base_variable.append(variable)
    base_variable = np.array(base_variable, dtype=float)
    return base_variable


def get_z_score_norm(train_base_variable, test_base_variable):
    """
    z-score标准化，
    这里应注意，在对测试集进行标准化时，用的也是通过训练集算得的均值和标准差"""
    mean = train_base_variable.mean(axis=0)  # 均值
    std = train_base_variable.std(axis=0)  # 标准差
    train_z_score_norm = (train_base_variable - mean) / std
    test_z_score_norm = (test_base_variable - mean) / std
    return train_z_score_norm, test_z_score_norm


def get_max_min_norm(train_base_variable, test_base_variable):
    """最大最小值归一化"""
    max_ = train_base_variable.max(axis=0)
    min_ = train_base_variable.min(axis=0)
    train_max_min_norm = (train_base_variable - min_) / (max_ - min_)
    test_max_min_norm = (test_base_variable - min_) / (max_ - min_)
    return train_max_min_norm, test_max_min_norm


def get_feature_variable(z_score_norm):
    """根据定义好的公式计算微博影响力和用户可信度"""
    feature_variable = []
    for ind in range(z_score_norm.shape[0]):
        time_interval = z_score_norm[ind][0]
        like = z_score_norm[ind][1]
        repost = z_score_norm[ind][2]
        comment = z_score_norm[ind][3]
        num = z_score_norm[ind][4]
        following = z_score_norm[ind][5]
        fans = z_score_norm[ind][6]
        if_verified = z_score_norm[ind][7]

        influence = (1. / time_interval) * (np.log(np.exp(fans) + np.exp(repost) + np.exp(like) + np.exp(comment)))
        reliability = np.log(np.exp(fans - following) + np.exp(num)) + if_verified

        # 2019年论文中的微博影响力
        old_influence = np.log(np.exp(fans) + np.exp(repost) + np.exp(comment)) + like

        # feature_variable.append((influence, reliability))
        feature_variable.append((old_influence, reliability))

    return feature_variable


def get_manual_features(config):
    """带入训练集和测试集进行计算，并保存至本地"""
    train_base_variable = get_base_variable(config.train_file_path)
    test_base_variable = get_base_variable(config.test_file_path)
    train_z_score_norm, test_z_score_norm = get_z_score_norm(train_base_variable, test_base_variable)
    train_feature_variable = get_feature_variable(train_z_score_norm)
    test_feature_variable = get_feature_variable(test_z_score_norm)
    train_manual_feature = np.array(train_feature_variable)
    test_manual_feature = np.array(test_feature_variable)

    # train_manual_feature_z, test_manual_feature_z = get_max_min_norm(train_manual_feature, test_manual_feature)
    # print(train_manual_feature)
    # print(test_manual_feature)
    np.save(config.train_manual_features_file_path, train_manual_feature)
    np.save(config.test_manual_features_file_path, test_manual_feature)
    print("手工特征已保存至本地。")


if __name__ == "__main__":
    model_config = ModelConfig()
    get_manual_features(model_config)
