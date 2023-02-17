import os
import logging
from gensim import corpora
from tqdm.auto import tqdm
import pandas as pd
from collections import defaultdict

from utils.data_helpers import Preprocessor, tokenize_ltp
from topic_config import TopicConfig
from lda import train_save_lda_model, infer_lda_topics, extract_topic_from_lda_prediction, load_lda_model, evaluate_lda_topic_model


# -----------------------------------------------
# Preprocessing function
# -----------------------------------------------

def lda_preprocess(data_tokenized, id2word=None, delete_stopwords=True, stopwords_file_path=None,
                   processeds_text_file_path=None, print_steps=False):
    '''
    Preprocess tokenized text data for LDA (deleting stopwords, recognising ngrams, lemmatisation
    :param data_tokenized: 分词后的数据
    :param id2word: preexisting id2word object (to map dev or test split to identical ids), otherwise None (train split)
    :param stopwords_file_path: 停用词的路径
    :param processeds_text_file_path: 去停用词后保存文件的路径
    :param print_steps: 是否打印出示例
    :return: preprocessed corpus as bag of wordids, id2word
    '''
    assert type(data_tokenized) == list
    assert type(data_tokenized[0]) == list
    assert type(data_tokenized[0][0]) == str

    data_finished = data_tokenized
    # 过滤字符长度为1的词
    data_finished = [Preprocessor.remove_short_words(s) for s in data_finished]

    if delete_stopwords:
        print('removing stopwords')
        data_finished = [Preprocessor.remove_stopwords(sentence, stopwords_file_path) for sentence in tqdm(data_finished, desc='Delete Stopwords')]
    if print_steps:
        print(data_finished[:1])

    # 过滤低频词 去掉只出现一次的单词
    frequency = defaultdict(int)
    for sentence in data_finished:
        for token in sentence:
            frequency[token] += 1
    data_filtered = [[token for token in sentence if frequency[token] > 1] for sentence in data_finished]

    if id2word is None:
        # Create Dictionary
        id2word = corpora.Dictionary(data_filtered)

        # 过滤极值（去掉频率占词典总数50%的，以及出现频次小于5的
        id2word.filter_extremes(no_below=3,  # 去掉出现次数低于no_below的
                                no_above=0.5,  # 去掉出现次数高于no_above的，注意这个小数指的是百分数
                                keep_n=50000  # 在前面两个参数的基础上，保留出现频率前keep_n的单词
                                )
        id2word.compactify()  # 在执行完过滤操作以后，会造成单词的序号之间有空隙，这时就可以使用该函数来对词典来进行重新排序，去掉这些空隙

    with open(processeds_text_file_path, 'a+', encoding='utf-8') as processed_file:
        for row in data_filtered:
            processed_file.write(str(row) + '\n')

    # Create Corpus
    processed_texts = data_filtered

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in processed_texts]

    # View
    if print_steps:
        # bag of ids
        print(corpus[20])
        # Human readable format of corpus (term-frequency)
        print(boids_to_human(corpus[:20], id2word))
    return corpus, id2word, processed_texts


def boids_to_human(corpus, id2word):
    '''
    Convenience function to display bag of words instead of bag of ids
    :param corpus: bag of word ids
    :param id2word: id to word mapping dict
    :return: bag of words
    '''
    if len(corpus[0]) > 0:
        # multiple sentences
        human_format = [[(id2word[id], freq) for id, freq in cp] for cp in corpus]
    else:
        # one sentence
        human_format = [(id2word[id], freq) for id, freq in corpus]
    return human_format

# -----------------------------------------------
# functions to train and save topic models
# -----------------------------------------------

def train_topic_model(corpus, id2word, topic_config, save_model=False):
    '''
    选择不同主题模型进行训练
    :param corpus: preprocessed corpus
    :param id2word: id2word from preprocessing step
    :param config: 超参数
    :return: trained topic model
    '''
    if not os.path.exists(topic_config.lda_model_save_dir):
        os.mkdir(topic_config.lda_model_save_dir)
    print('Training {} model...'.format(topic_config.topic_type))
    if topic_config.topic_type == 'LDA':
        return train_save_lda_model(corpus, id2word, topic_config, save_model)
    elif topic_config.topic_type == 'other_topic_model':
        pass
    else:
        raise ValueError('topic_type should be "LDA" or "other topic model".')

def infer_topic_dist(new_corpus, new_processed_texts, topic_model, topic_type):
    '''
    Get global topic distribution of all sentences from dev or test set from lda_model
    '''
    print('Inferring topic distribution...')
    # obtain topic distribution for dev set
    assert type(new_processed_texts) == list # of sentences
    assert type(new_processed_texts[0]) == list # of tokens
    assert type(new_processed_texts[0][0]) == str
    assert type(new_corpus) == list  # of sentences
    assert type(new_corpus[0]) == list  # of word id/count tuples
    assert type(new_corpus[0][0]) == tuple

    global_topics = None
    if topic_type == 'LDA':
        dist_over_topic = infer_lda_topics(topic_model, new_corpus)
        # extract topic vectors from prediction
        # global_topics = extract_topics_from_prediction(dist_over_topic, topic_type, TopicModel.num_topics)
        global_topics = extract_topic_from_lda_prediction(dist_over_topic, topic_model.num_topics)
    elif topic_type == 'other_topic_model':
        pass
    # sanity check
    assert (len(global_topics.shape) == 2)
    assert (global_topics.shape[1] == topic_model.num_topics)
    print(global_topics.shape)
    print('topic distribution obtain completely.')
    return global_topics

def extract_topics_from_prediction(dist_over_topic, type, num_topics):
    '''
    Selects the correct topic extraction function based on topic model type
    :param dist_over_topic: topic inference result from lda_model[new_corpus]
    :param type: 'LDA' or 'ldamallet'
    :return: topic array with (examples, num_topics)
    '''
    if type == 'LDA':
        return extract_topic_from_lda_prediction(dist_over_topic, num_topics)
    elif type == 'ldamallet':
        pass
        # return extract_topic_from_ldamallet_prediction(dist_over_topic)
    else:
        raise ValueError('Incorrect topic type: {}. Should be "LDA" or "ldamallet"'.format(type))


# -----------------------------------------------
# convenience functions
# -----------------------------------------------

def load_topic_model(topic_config):
    topic_model = None
    if topic_config.topic_type == 'LDA':
        topic_model = load_lda_model(topic_config)
    elif topic_config.topic_type == 'other_topic_model':
        pass
    return topic_model


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

    # 训练主题模型时，只用训练集数据
    data_tokenized = tokenize_ltp(sentences, user_dict_filepath=topic_config.user_dict_file_path,
                                  filepath=topic_config.segmented_file_path, postfix=topic_config.tokenize_type)
    corpus, id2word, processed_texts = lda_preprocess(data_tokenized, id2word=None, delete_stopwords=True,
                                                      processeds_text_file_path=topic_config.processed_file_path,
                                                      stopwords_file_path=topic_config.stopwords_file_path,
                                                      print_steps=True)

    # 根据得到的最优主题数目训练模型
    # topic_model_gensim = train_topic_model(corpus, id2word, topic_config, save_model=True)
    # 加载训练好的lda模型
    topic_model = load_topic_model(topic_config)
    print(topic_model.num_topics)

    # 下一步：写文档-主题分布和主题-词分布





