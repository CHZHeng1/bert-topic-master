import logging

from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from topic_trainer import train_topic_model, lda_preprocess
from lda import evaluate_lda_topic_model
from topic_config import TopicConfig
from utils.data_helpers import Preprocessor, tokenize_ltp, tokenize_jieba


def get_topic_num(corpus, topic_config, topic_range):
    """通过困惑度和主题一致性确定最优主题个数（暴力搜索）"""
    perplexity = []
    for num_topic in tqdm(topic_range, desc='access topic number processing'):
        topic_config.num_topics = num_topic
        lda_train_model = train_topic_model(corpus, topic_config, save_model=False)
        model_perplexity = evaluate_lda_topic_model(lda_train_model, corpus)
        perplexity.append(model_perplexity)

    better_num_topic_from_perplexity = topic_range[perplexity.index(min(perplexity))]

    logging.info(f"### perplexity: {perplexity} ")
    logging.info(f'### the number of topic is {better_num_topic_from_perplexity}, the perplexity is {min(perplexity)} (lower)')
    return perplexity


def draw_perplexity_plot(topic_range, perplexity, topic_config):
    """
    绘制主题一致性随主题个数变化趋势图
    param: topic_range 主题个数范围
    param: coherence 主题个数对应的主题一致性得分
    """
    X = topic_range
    y = perplexity
    plt.style.use('seaborn-whitegrid')  # 设置图像的风格
    sns.set_style("white")
    plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决画图时中文乱码问题
    plt.figure(figsize=(6, 4))  # 设置画布大小
    plt.title("topics-perplexity")  # 设置图像标题
    plt.plot(X, y, c='r', marker='o')
    plt.xlabel('topics')
    plt.ylabel('log_perplexity')
    plt.grid(alpha=.4, axis="y")  # 显示背景中的网格
    plt.gca().spines["top"].set_alpha(.0)  # 让上方和右侧的坐标轴被隐藏
    plt.gca().spines["right"].set_alpha(.0)
    plt.savefig(f'{topic_config.lda_results_save_dir}\\topics_perplexity.jpg', dpi=300)
    # plt.show()


if __name__ == '__main__':
    topic_config = TopicConfig()
    train_file_path = topic_config.train_file_path
    raw_iter = pd.read_csv(train_file_path)
    sentences = []
    for raw in tqdm(raw_iter.values, desc='Data Processing:', ncols=80):  # ncols: 进度条长度
        sentence = raw[1]  # 文本
        sentence_1 = Preprocessor.basic_pipeline(sentence)  # 替换文本中的url、@id、繁体转简体
        sentence_2 = Preprocessor.process_for_segmented(sentence_1)
        sentences.append(sentence_2)

    # ltp分词
    data_tokenized = tokenize_ltp(sentences, user_dict_filepath=topic_config.user_dict_file_path,
                                  filepath=topic_config.segmented_file_path, postfix=topic_config.tokenize_type)
    # jieba分词
    # data_tokenized = tokenize_jieba(sentences, user_dict_filepath=topic_config.user_dict_file_path,
    #                                 filepath=topic_config.segmented_file_path, postfix=topic_config.tokenize_type)

    corpus, id2word, processed_texts = lda_preprocess(data_tokenized, id2word=None, delete_stopwords=True,
                                                      stopwords_file_path=topic_config.stopwords_file_path,
                                                      processed_text_file_path=topic_config.processed_file_path,
                                                      print_steps=True)

    # 确定主题个数
    # topic_range = [t * 5 for t in range(1, 11)]  # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    topic_range = list(range(2, 31))
    # 确定最优主题个数后, 需返回TopicConfig.py中修改 self.num_topics 的值
    perplexity = get_topic_num(corpus, topic_config, topic_range)
    draw_perplexity_plot(topic_range, perplexity, topic_config)

