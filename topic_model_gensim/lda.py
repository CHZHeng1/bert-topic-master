import os
import numpy as np
import gensim
from gensim.models import CoherenceModel
from topic_config import TopicConfig

# -----------------------------------------------
# save and load functions for lda topic models
# -----------------------------------------------

def train_save_lda_model(corpus, id2word, topic_config, save_model=False):
    '''
    基于gensim训练并保存lda模型
    :param corpus: preprocessed corpus
    :param id2word: id2word from preprocessing step
    :return: trained topic model
    '''
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=topic_config.num_topics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    # save
    if save_model:
        save_topic_model(lda_model, topic_config)
    return lda_model

def save_topic_model(topic_model, topic_config):
    '''保存训练好的模型'''
    model_path = os.path.join(topic_config.lda_model_save_dir, 'lda_model')
    print('Saving Topic Model to {}...'.format(model_path))
    topic_model.save(model_path)

def load_lda_model(config):
    model_path = os.path.join(config.lda_model_save_dir, 'lda_model')
    print('Loading Topic Model from {}...'.format(model_path))
    lda_model = gensim.models.LdaModel.load(model_path)
    print('Done.')
    return lda_model

# -----------------------------------------------
# functions to infer topic distribution
# -----------------------------------------------
def infer_lda_topics(lda_model, new_corpus):
    '''推断主题概率分布'''
    assert type(new_corpus) == list # of sentences
    assert type(new_corpus[0]) == list # of word id/count tuples
    assert type(new_corpus[0][0]) == tuple
    return lda_model[new_corpus]

# -----------------------------------------------
# functions to explore topic distribution
# -----------------------------------------------
def extract_topic_from_lda_prediction(dist_over_topic, num_topics):
    '''
    获取文档-主题分布 -> shape(docs, num_topics)
    :param dist_over_topic: 通过lda模型训练获取的主题分布对象 -> dist_over_topic = lda_model[corpus]
    :param num_topics: 主题个数
    :return: topic array with (examples, num_topics)
    '''
    # 遍历嵌套表示并将主题分布提取为单个向量，每个文档的长度=num_topics
    topic_array = []
    for j, example in enumerate(dist_over_topic):
        i = 0
        topics_per_doc = []
        for topic_num, prop_topic in example[0]:
            # print(i)
            # print(topic_num)
            while not i == topic_num:
                topics_per_doc.append(0.)  # 用0.填充缺失主题概率
                # print('missing')
                i = i + 1
            topics_per_doc.append(prop_topic)
            i = i + 1
        while len(topics_per_doc) < num_topics:
            topics_per_doc.append(0.)  # 填充最后缺失的主题概率 如共10个主题，但只有0,1,2...,7的主题概率分布,则8和9的填充为0.
        topic_array.append(np.array(topics_per_doc))
    global_topics = np.array(topic_array)
    # sanity check
    if not ((len(global_topics.shape) == 2) and (global_topics.shape[1] == num_topics)):
        print('Inconsistent topic vector length detected:')
        i = 0
        for dist, example in zip(global_topics, dist_over_topic):
            if len(dist) != num_topics:
                # print('{}th example with length {}: {}'.format(i, len(dist), dist))
                print(f'{i}th example with length {len(dist)}: {dist}')
                # print('from: {}'.format(example[0]))
                print(f'from: {example[0]}')
                print('--')
            i += 1
    return global_topics

def evaluate_lda_topic_model(lda_model, corpus, processed_texts, id2word):
    '''
    通过计算困惑度和主题一致性来进行内部评价
    :param lda_model:
    :param corpus:
    :param processed_texts:
    :param id2word:
    :return: dictionary with perplexity and coherence
    '''
    model_perplexity = None
    try:
        model_perplexity = lda_model.log_perplexity(corpus)
        print('\nPerplexity: ', model_perplexity)  # a measure of how good the model is. lower the better.
        # results = {'perplexity': model_perplexity}
    except AttributeError:
        results = {}
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_texts, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()  # the higher the better
    print('\nCoherence Score: ', coherence_lda)
    # results['coherence'] = coherence_lda
    return model_perplexity, coherence_lda

if __name__ == '__main__':
    config = TopicConfig()
    model_path = os.path.join(config.lda_model_save_dir, 'lda_model')
    print(model_path)
