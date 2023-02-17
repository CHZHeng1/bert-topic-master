# 此模块用于进行数据集的划分，完成后返回../data/文件夹目录下可以看到生成的数据划分后的train、test、val
import pandas as pd


def data_type_split(data_file_path):
    """
    根据标签划分数据类型，虚假信息、非虚假信息
    :param data_file_path: 需要划分数据类型的数据集（csv格式的文件） -> str
    :return: 根据不同标签类别切分好的数据 -> DataFrame
    """
    total_data = pd.read_csv(data_file_path)
    total_data['微博发布时间'] = pd.to_datetime(total_data['微博发布时间'])  # 改为时间格式
    total_data = total_data.sort_values(by='微博发布时间', ascending=True)  # 按照时间升序排列
    fake_news = total_data[total_data['label'] == 1]  # 720
    real_news = total_data[total_data['label'] == 0]  # 936
    fake_news.index = range(fake_news.shape[0])  # 恢复索引
    real_news.index = range(real_news.shape[0])
    return fake_news, real_news


def data_split(data):
    """
    数据集划分函数，按照7:2:1的比例划分训练集、测试集、验证集
    注意：这里不再随机切分数据集(随机切分有可能会效果特别好，从而导致结果没有区分度)，而是按照顺序进行划分
    :param data: 要进行切分的数据集 -> DataFrame
    :return: 切分好的数据集
    """
    num_train = int(len(data) * 0.7)
    num_val = int(len(data) * 0.1)
    num_test = len(data) - num_train - num_val
    train_data = data.iloc[:num_train, :]
    val_data = data.iloc[num_train:num_train+num_val, :]
    test_data = data.iloc[-num_test:, :]
    return train_data, val_data, test_data


def data_concat(real_news, fake_news, subset_name):
    """
    将划分好的不同类别的train、test、val合并在一起
    :param real_news: 非虚假信息 -> DataFrame
    :param fake_news: 虚假信息 -> DataFrame
    :param subset_name: 数据集名称 -> str
    :return: 合并完成的数据 -> DataFrame
    """
    subset_data = pd.concat([real_news, fake_news], axis=0)
    subset_data.index = range(subset_data.shape[0])
    subset_data.to_csv(f'../data/{subset_name}.csv', index=False, encoding='utf-8-sig')
    return subset_data


def sample_count(data):
    """观察划分后的训练集和测试集中正负样本的比例"""
    pos = 0
    for index in data.values:
        label = index[-1]
        if int(label) == 1:
            pos += 1
    neg = len(data) - pos
    return pos, neg


if __name__ == '__main__':
    """
    整体思路：根据现有数据集按照时间顺序（升序）划分train、test、val
    1.首先根据标签将不同类别的数据进行划分
    2.然后根据不同类别的数据分别按照7:2:1的比例划分train、test、val
    3.将划分好的不同类别的数据集按照train、test、val进行合并
    """
    # 按标签分类
    data_file_path = '../data/fake_news.csv'
    fake_news, real_news = data_type_split(data_file_path)
    fake_news_train, fake_news_val, fake_news_test = data_split(fake_news)
    real_news_train, real_news_val, real_news_test = data_split(real_news)
    # 数据集划分
    train_data = data_concat(real_news_train, fake_news_train, 'train')
    val_data = data_concat(real_news_val, fake_news_val, 'val')
    test_data = data_concat(real_news_test, fake_news_test, 'test')
    # 统计正负样本个数
    train_pos, train_neg = sample_count(train_data)
    val_pos, val_neg = sample_count(val_data)
    test_pos, test_neg = sample_count(test_data)

    print(f'训练集中共有样本{train_data.shape[0]}条，其中，正样本为{train_pos}条，负样本为{train_neg}条。')
    print(f'验证集中共有样本{val_data.shape[0]}条，其中，正样本为{val_pos}条，负样本为{val_neg}条。')
    print(f'测试集中共有样本{test_data.shape[0]}条，其中，正样本为{test_pos}条，负样本为{test_neg}条。')



