import os
import numpy as np


# -----------------------------------------------
# prediction loaders
# -----------------------------------------------
def load_document_topics_(topic_config, recover_topic_peaks=False, max_m=None):
    '''
    Loads inferred topic distribution for each sentence in dataset. Dependent on data subset.
    :param recover_topic_peaks: 将文档主题分布中最小的主题分布概率赋值为0, type -> True or False.
    :param max_m: 文档最大长度, 为减少文档数量.
    :return: document topics in list corresponding to topic_dists.
    '''
    # datasets, subsets = load_original_data(topic_config)
    # set model paths
    filepaths = []
    topic_model_folder = topic_config.lda_doc_topics_distribution_save_dir
    subsets_name = ['train', 'test']
    for ind, subset in enumerate(subsets_name):  # train, dev, test
        filepaths.append(os.path.join(topic_model_folder, subsets_name[ind] + '.npy'))
    # load
    topic_dists = [np.load(f) for f in filepaths]  # 文档主题分布 -> [train, dev, test]

    if recover_topic_peaks:
        for split in range(len(topic_dists)):
            for line in range(len(topic_dists[split])):
                topic_dists[split][line] = unflatten_topic(topic_dists[split][line])

    # reduce number of examples if necessary
    # max_examples = opt.get('max_m',None)
    if not max_m is None:
        topic_dists = [t[:max_m] for t in topic_dists]
    return topic_dists


def load_document_topics(document_topics_file_path, recover_topic_peaks=False, max_m=None):
    '''
    Loads inferred topic distribution for each sentence in dataset. Dependent on data subset.
    :param recover_topic_peaks: 将文档主题分布中最小的主题分布概率赋值为0, type -> True or False.
    :param max_m: 文档最大长度, 为减少文档数量.
    :return: document topics in list corresponding to topic_dists.
    '''
    # load
    topic_dists = np.load(document_topics_file_path)

    if recover_topic_peaks:
        for line in range(len(topic_dists)):
            topic_dists[line] = unflatten_topic(topic_dists[line])

    # reduce number of examples if necessary
    # max_examples = opt.get('max_m',None)
    if not max_m is None:
        topic_dists = topic_dists[:max_m]
    return topic_dists


def unflatten_topic(topic_vector):
    '''将每一篇文档主题分布中的最小主题概率分布赋值为0, 考虑是减小噪声'''
    # unflatten topic distribution
    min_val = topic_vector.min()
    for j, topic in enumerate(topic_vector):
        if topic == min_val:
            topic_vector[j] = 0
    return topic_vector


def load_word_topics_(topic_config, add_unk = True, recover_topic_peaks=False):
    '''
    Reads word topic vector and dictionary from file
    :param opt: option dictionary containing settings for topic model
    :return: word_topic_dict,id_word_dict
    '''
    complete_word_topic_dict = {}
    id_word_dict = {}
    word_id_dict = {}
    count = 0
    topic_matrix = []
    num_topics = topic_config.num_topics

    unk_topic = topic_config.unk_topic

    word_topics_distribution_folder = topic_config.lda_word_topics_distribution_save_dir
    word_topic_file = os.path.join(word_topics_distribution_folder, 'word_topics.log')
    # todo:train topic model
    print("Reading word_topic vector from {}".format(word_topic_file))
    if add_unk:
        # add line for UNK word topics 未登录词的处理
        word = '<nontopic>'
        if unk_topic == 'zero':
            topic_vector = np.array([0.0]*num_topics)
        elif unk_topic == 'uniform':
            assert not recover_topic_peaks, "Do not use unk_topic='uniform' and 'unflat_topics'=True' together. " \
                                            "As it will result in flattened non-topics, but unflattened topics."
            topic_vector = np.array([1/num_topics] * num_topics)
        else:
            raise ValueError
        wordid = 0
        id_word_dict[wordid] = word
        word_id_dict[word] = wordid
        complete_word_topic_dict[word] = topic_vector
        topic_matrix.append(topic_vector)
    # read other word topics
    with open(word_topic_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            count += 1
            if count > 2:
                ldata = line.rstrip().split(' ')  # \xa in string caused trouble
                if add_unk:
                    wordid = int(ldata[0])+1  # 索引号
                else:
                    wordid = int(ldata[0])
                word = ldata[1]  # 词
                id_word_dict[wordid] = word  # {id:word,...}
                word_id_dict[word] = wordid  # {word:id,...}
                # print(ldata[2:])
                topic_vector = np.array([float(s.replace('[', '').replace(']', '')) for s in ldata[2:]])
                assert len(topic_vector) == num_topics
                if recover_topic_peaks:
                    topic_vector = unflatten_topic(topic_vector)
                complete_word_topic_dict[word] = topic_vector  # {word:array([xxx,xxx,xxx,...])}
                topic_matrix.append(topic_vector)
    topic_matrix = np.array(topic_matrix)  # array([[xxx,xxx,xxx,...],[xxx,xxx,xxx,...]...])
    print('word topic embedding dim: {}'.format(topic_matrix.shape))
    assert len(topic_matrix.shape) == 2
    return {'complete_topic_dict': complete_word_topic_dict, 'topic_dict': id_word_dict,
            'word_id_dict': word_id_dict, 'topic_matrix': topic_matrix}


def load_word_topics(config, add_unk = True, recover_topic_peaks=False):
    """
    Reads word topic vector and dictionary from file
    :param config: option class containing settings for topic model
    :return: word_topic_dict,id_word_dict
    """
    complete_word_topic_dict = {}
    id_word_dict = {}
    word_id_dict = {}
    count = 0
    topic_matrix = []
    num_topics = config.num_topics
    unk_topic = config.unk_topic
    word_topic_file = config.word_topics_file_path
    print("Reading word_topic vector from {}".format(word_topic_file))

    if add_unk:
        # add line for UNK word topics 未登录词的处理
        word = '<nontopic>'
        if unk_topic == 'zero':
            topic_vector = np.array([0.0]*num_topics)
        elif unk_topic == 'uniform':
            assert not recover_topic_peaks, "Do not use unk_topic='uniform' and 'unflat_topics'=True' together. " \
                                            "As it will result in flattened non-topics, but unflattened topics."
            topic_vector = np.array([1/num_topics] * num_topics)
        else:
            raise ValueError
        wordid = 0
        id_word_dict[wordid] = word
        word_id_dict[word] = wordid
        complete_word_topic_dict[word] = topic_vector
        topic_matrix.append(topic_vector)
    # read other word topics
    with open(word_topic_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            count += 1
            if count > 2:
                ldata = line.rstrip().split(' ')  # \xa in string caused trouble
                if add_unk:
                    wordid = int(ldata[0])+1  # 索引号
                else:
                    wordid = int(ldata[0])
                word = ldata[1]  # 词
                id_word_dict[wordid] = word  # {id:word,...}
                word_id_dict[word] = wordid  # {word:id,...}
                # print(ldata[2:])
                topic_vector = np.array([float(s.replace('[', '').replace(']', '')) for s in ldata[2:]])
                assert len(topic_vector) == num_topics
                if recover_topic_peaks:
                    topic_vector = unflatten_topic(topic_vector)
                complete_word_topic_dict[word] = topic_vector  # {word:array([xxx,xxx,xxx,...])}
                topic_matrix.append(topic_vector)
    topic_matrix = np.array(topic_matrix)  # array([[xxx,xxx,xxx,...],[xxx,xxx,xxx,...]...])
    print('word topic embedding dim: {}'.format(topic_matrix.shape))
    assert len(topic_matrix.shape) == 2
    return {'complete_topic_dict': complete_word_topic_dict, 'topic_dict': id_word_dict,
            'word_id_dict': word_id_dict, 'topic_matrix': topic_matrix}


if __name__ == '__main__':
    from config import ModelConfig
    config = ModelConfig()
    # topic_dists = load_document_topics(config.train_document_topics_file_path)
    # print(type(topic_dists))
    # print(len(topic_dists))
    # print(topic_dists.shape)
    word_topic_dict = load_word_topics(config,)
    # print(word_topic_dict['complete_topic_dict'])
    # print(word_topic_dict['topic_dict'])
    # print(word_topic_dict['word_id_dict'])
    print((word_topic_dict['topic_matrix']).shape)

