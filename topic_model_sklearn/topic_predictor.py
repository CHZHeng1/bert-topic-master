import os
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from topic_trainer import load_topic_model, lda_preprocess, infer_topic_dist, train_topic_model
from utils.data_helpers import Preprocessor, tokenize_ltp

# -----------------------------------------------
# infer, save, load topic distribution functions for dataset
# -----------------------------------------------

# --------  document topic model --------


def infer_and_write_document_topics(topic_config, topic_model=None, id2word=None):
    '''
    Infer global topic distribution for all documents in data splits (e.g. train, dev, test) mentioned in opt['subsets']
    and save as for both documents separately as numpy arrays. 获取所有文档的全局主题分布(包含训练集、验证集和测试集)
    :return: Nothing
    '''
    train_raw_iter = pd.read_csv(topic_config.train_file_path)
    test_raw_iter = pd.read_csv(topic_config.test_file_path)
    subsets = [train_raw_iter, test_raw_iter]

    # try to load topic model
    if topic_model is None:
        topic_model = load_topic_model(topic_config)
    if id2word is None:
        raise ValueError('id2word is should be not none')

    topic_dist_list = []  # [topic_dist_train, topic_dist_dev, topic_dist_test]

    subsets_name = ['train', 'test']
    for ind, subset in enumerate(subsets):  # subsets -> [train, dev, test]
        # preprocess and infer topics
        sent = []
        for raw in tqdm(subset.values, desc='Data Processing'):
            s = raw[1]  # 文本
            s = Preprocessor.basic_pipeline(s)  # 替换文本中的url、@id、繁体转简体
            s = Preprocessor.process_for_segmented(s)
            sent.append(s)

        # set segmented file path  -> segmented_train_len(train_data).pt
        segmented_file_path = os.path.join(topic_config.segmented_data_save_dir, 'segmented_'+subsets_name[ind]+'.pt')
        processed_file_path = os.path.join(topic_config.segmented_data_save_dir,
                                           'processed_text_'+subsets_name[ind]+'.txt')
        # sanity check
        assert len(sent) == len(subset)

        sent_tokenized = tokenize_ltp(sent, user_dict_filepath=topic_config.user_dict_file_path,
                                      filepath=segmented_file_path, postfix=topic_config.tokenize_type)

        new_corpus, _, new_processed_texts = lda_preprocess(sent_tokenized, id2word=id2word, delete_stopwords=True,
                                                            processed_text_file_path=processed_file_path,
                                                            stopwords_file_path=topic_config.stopwords_file_path,
                                                            print_steps=False)
        # topic_dist.shape -> (examples, num_topics)
        topic_dist = infer_topic_dist(new_corpus, topic_model, topic_config.topic_type)
        assert len(new_processed_texts) == topic_dist.shape[0]

        # set model path
        topic_model_folder = topic_config.lda_doc_topics_distribution_save_dir

        # make folder if not existing
        if not os.path.exists(topic_model_folder):
            os.mkdir(topic_model_folder)

        topic_dist_path = os.path.join(topic_model_folder, subsets_name[ind])  # train.npy test.npy

        # save as separate numpy arrays
        np.save(topic_dist_path, topic_dist)

        topic_dist_list.extend([topic_dist])
    return topic_dist_list


# --------  word topic model --------


def infer_and_write_word_topics(topic_config, topic_model=None, id2word=None, max_vocab=None):
    """
    Loads a topic model and writes topic predictions for each word in dictionary to file.
    :param id2word: id2word = vector_model.fit(data_finished)
    :param max_vocab: yourself decided
    :return: void
    """
    if topic_model is None or id2word is None:
        raise ValueError('id2word is should be not none')

    print('Infering and writing word topic distribution ...')
    if max_vocab is None:
        vocab_len = len(id2word.get_feature_names_out())
    else:
        vocab_len = max_vocab

    # create one word documents in bag of words format for topic model
    # 为主题模型创建词袋格式的单词文档
    print(vocab_len)

    # word_topics.shape -> (num_words, num_topics)
    word_topics = topic_model.components_.T

    word_topics_distribution_folder = topic_config.lda_word_topics_distribution_save_dir
    # make folder if not existing
    if not os.path.exists(word_topics_distribution_folder):
        os.mkdir(word_topics_distribution_folder)
    word_topic_file = os.path.join(word_topics_distribution_folder, 'word_topics.log')
    with open(word_topic_file, 'w', encoding='utf-8') as outfile:
        # write to file
        model_path = os.path.join(topic_config.lda_model_save_dir, 'lda_model.pkl')
        outfile.writelines('Loading Topic Model from {}\n'.format(model_path))
        outfile.writelines('{}\n'.format(vocab_len))
        for i in range(vocab_len):
            # for i,(k,w) in enumerate(topic_model.id2word.items()):
            w = id2word.get_feature_names_out()[i]  # token
            if max_vocab is None or i < max_vocab:
                # replace multiple spaces and new line
                # \s: 匹配任何空白字符,等价于[ \f\n\r\t\v] +: 匹配前面的子表达式一次或多次
                vector_str = re.sub('\s+', ' ', str(word_topics[i]))
                # remove space for last element in case of 'short' float
                # 去除 ] 前的空白字符
                vector_str = re.sub('\s+]', ']', str(vector_str))
                vector_str = vector_str.replace('[ ', '[')
                line = '{} {} {}'.format(i, w, vector_str)
                # print(line)
                outfile.writelines(line+'\n')
            else:
                break
    print(f'word topics distribution saved to {word_topic_file}.')


if __name__ == '__main__':
    from topic_config import TopicConfig
    topic_config = TopicConfig()
    train_file_path = topic_config.train_file_path
    raw_iter = pd.read_csv(train_file_path)
    sentences = []
    for raw in tqdm(raw_iter.values, desc='Data Processing'):
        sentence = raw[1]  # 文本
        sentence_1 = Preprocessor.basic_pipeline(sentence)  # 替换文本中的url、@id、繁体转简体
        sentence_2 = Preprocessor.process_for_segmented(sentence_1)
        sentences.append(sentence_2)

    # 训练主题模型时，只用训练集数据
    data_tokenized = tokenize_ltp(sentences, user_dict_filepath=topic_config.user_dict_file_path,
                                  filepath=topic_config.segmented_file_path, postfix=topic_config.tokenize_type)
    corpus, id2word, processed_texts = lda_preprocess(data_tokenized, id2word=None, delete_stopwords=True,
                                                      stopwords_file_path=topic_config.stopwords_file_path,
                                                      processed_text_file_path=topic_config.processed_file_path,
                                                      print_steps=False)

    topic_model = train_topic_model(corpus, topic_config, save_model=True)
    infer_and_write_document_topics(topic_config, topic_model=topic_model, id2word=id2word)
    infer_and_write_word_topics(topic_config, topic_model=topic_model, id2word=id2word, max_vocab=None)


