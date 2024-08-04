import os
import logging
import re

import jieba
import numpy as np
import pandas as pd
from hanziconv import HanziConv
from ltp import LTP
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from topic_model.topic_loader import load_document_topics, load_word_topics


def cache(func):
    """
    本修饰器的作用是data_process()方法处理后的结果进行缓存，下次使用时可直接载入！
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        filepath = kwargs['filepath']
        postfix = kwargs['postfix']
        data_path = filepath.rsplit('.', 1)[0] + '_' + postfix + '.pt'
        # data_path = filepath + '\\data_cache' + '_' + postfix + '.pt'
        if not os.path.exists(data_path):
            logging.info(f"缓存文件 {data_path} 不存在，重新处理并缓存！")
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            logging.info(f"缓存文件 {data_path} 存在，直接载入缓存文件！")
            with open(data_path, 'rb') as f:
                data = torch.load(f)
        return data

    return wrapper


class Preprocessor:
    """数据处理模块"""
    vocab_processor = None

    @staticmethod
    def basic_pipeline(s):
        """
        基本数据清洗，bert和ltp分词的必要步骤
        :param s: 一条样本 s -> str
        """
        # process text
        # print("Preprocessor: replace urls,@id,invalid...")
        s = Preprocessor.replace_url_id_invalid(s)
        # print("Preprocessor: traditional to simplified")
        s = Preprocessor.traditional_to_simplified(s)
        return s

    @staticmethod
    def replace_url_id_invalid(s):
        """
        替换url和@id
        :param s: 一条样本
        """
        url_token = '<URL>'
        img_token = '<IMG>'
        at_token = '<@ID>'
        # s = re.sub(r'(http://)?www.*?(\s|$|[\u4e00-\u9fa5])', url_token + '\\2', s)  # URL containing www
        # s = re.sub(r'http://.*?(\s|$|[\u4e00-\u9fa5])', url_token + '\\1', s)  # URL starting with http
        s = re.sub(r'(http://)?www.*?(\s|$)', url_token + '\\2', s)
        s = re.sub(r'http://.*?(\s|$)', url_token + '\\1', s)
        s = re.sub(r'\w+?@.+?\\.com.*', url_token, s)  # email
        s = re.sub(r'\[img.*?\]', img_token, s)  # image
        s = re.sub(r'< ?img.*?>', img_token, s)
        s = re.sub(r'@.*?(\s|$|：)', at_token + '\\1', s)  # @id...
        s = re.sub('\u200B', '', s)
        s = s.strip()
        return s

    @staticmethod
    def traditional_to_simplified(s):
        """繁体转简体"""
        return HanziConv.toSimplified(s.strip())

    @staticmethod
    def process_for_segmented(s):
        """数据处理,用于ltp分词"""
        s = re.sub(r'<@ID>|<URL>', '', s)
        s = re.sub(r'\s+', '，', s.strip())  # 将空白字符替换成逗号
        # s = re.sub(r'\W', '', s.strip())  # 过滤所有非字母数字下划线的字符
        return s

    @staticmethod
    def stop_words_list(stopwords_file_path):
        """加载停用词表"""
        stopwords = [line.strip() for line in open(stopwords_file_path, 'r', encoding='utf-8').readlines()]
        return stopwords

    @staticmethod
    def remove_stopwords(sentence, stopwords_file_path):
        stopwords = Preprocessor.stop_words_list(stopwords_file_path)
        return [i for i in sentence if i not in stopwords]

    @staticmethod
    def remove_short_words(s):
        """过滤字符长度为1的词"""
        return [w for w in s if len(w) > 1]

    @staticmethod
    def remove_special_words(sentence):
        """过滤非字母数字下划线的字符"""
        return [re.sub('\W', '', w) for w in sentence]

    @staticmethod
    def ltp_init(user_dict_filepath):
        """
        初始化ltp
        自定义词典 也可以使用
        ltp.init_dict(path="user_dict.txt", max_window=4)
        user_dict.txt 是词典文件， max_window是最大前向分词窗口 详见：https://ltp.ai/docs/quickstart.html#id6
        """
        ltp = LTP()
        ltp.init_dict(path=user_dict_filepath, max_window=4)
        # user_dict = ['新冠', '疫情', '90后', '00后', 'MU5735', '东航', '瑞金医院']
        # ltp.add_words(words=user_dict, max_window=4)
        return ltp

    @staticmethod
    def jieba_init(user_dict_filepath):
        """初始化结巴分词工具，加载自定义词典"""
        return jieba.load_userdict(user_dict_filepath)


@cache
def tokenize_ltp(sentences, user_dict_filepath='./', filepath='./', postfix='cache'):
    """
    ltp中文分词
    sentences: 要进行分词的句子列表  -> list
    sentence: 要进行分词的句子 ->str
    """
    assert type(sentences) == list
    logging.info('## 正在使用ltp分词工具进行分词')
    ltp = Preprocessor.ltp_init(user_dict_filepath)  # 加载自定义词典
    result = []  # 分词结果
    for sentence in tqdm(sentences, desc='seg processing'):
        segment, _ = ltp.seg([sentence])
        # with open(filepath, 'a+', encoding='utf-8') as seg_file:
        #     seg_file.write(str(segment[0]) + '\n')
        result.append(segment[0])
    return result


@cache
def tokenize_jieba(sentences, user_dict_filepath='./', filepath='./', postfix='cache'):
    """jieba分词"""
    assert type(sentences) == list
    logging.info('## 正在使用jieba分词工具进行分词')
    result = []
    Preprocessor.jieba_init(user_dict_filepath)
    for sentence in tqdm(sentences, desc='seg processing'):
        segment_jieba = jieba.cut(sentence)
        segment_jieba = [token for token in segment_jieba]
        # with open(filepath, 'a+', encoding='utf-8') as seg_file:
        #     seg_file.write(str(segment_jieba) + '\n')
        result.append(segment_jieba)
    return result


class Vocab:
    """
    根据本地的vocab文件，构造一个词表
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    print(len(vocab))  # 返回词表长度
    """
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def build_vocab(vocab_path):
    """
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    """
    return Vocab(vocab_path)


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len :
                当max_len = 50时，表示以某个固定长度对样本进行padding，多余的截掉；
                当max_len=None时，表示以当前batch中最长样本的长度对其它进行padding；
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


class LoadSingleSentenceClassificationDataset:
    def __init__(self, vocab_path='./vocab.txt', tokenizer=None, batch_size=32, max_sen_len=None, split_sep='\n',
                 max_position_embeddings=512, pad_index=0, is_sample_shuffle=True):
        """
        :param vocab_path: 本地词表vocab.txt的路径
        :param tokenizer:
        :param batch_size:
        :param max_sen_len: 在对每个batch进行处理时的配置；
                            当max_sen_len = None时，即以每个batch中最长样本长度为标准，对其它进行padding
                            当max_sen_len = 'same'时，以整个数据集中最长样本为标准，对其它进行padding
                            当max_sen_len = 50， 表示以某个固定长度符样本进行padding，多余的截掉；
        :param split_sep: 文本和标签之前的分隔符，默认为'\t'
        :param max_position_embeddings: 指定最大样本长度，超过这个长度的部分将本截取掉
        :param is_sample_shuffle: 是否打乱训练集样本（只针对训练集）
                在后续构造DataLoader时，验证集和测试集均指定为了固定顺序（即不进行打乱），修改程序时请勿进行打乱
                因为当shuffle为True时，每次通过for循环遍历data_iter时样本的顺序都不一样，这会导致在模型预测时
                返回的标签顺序与原始的顺序不一样，不方便处理。
         """
        self.tokenizer = tokenizer
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']  # 句子分割符号索引
        self.CLS_IDX = self.vocab['[CLS]']  # 开始符号索引
        self.batch_size = batch_size
        self.split_sep = split_sep
        self.max_position_embeddings = max_position_embeddings
        if isinstance(max_sen_len, int) and max_sen_len > max_position_embeddings:
            max_sen_len = max_position_embeddings
        self.max_sen_len = max_sen_len
        self.is_sample_shuffle = is_sample_shuffle

    # BERT
    @cache
    def data_process(self, filepath, postfix='cache'):
        """
        在得到构建的字典后，便可以通过如下函数来将训练集、验证集和测试集转换成Token序列
        将每一句话中的每一个词根据字典转换成索引的形式，同时返回所有样本中最长样本的长度
        :param filepath: 数据集路径
        :return:
        """

        raw_iter = pd.read_csv(filepath)
        data = []
        max_len = 0
        for raw in tqdm(raw_iter.values, desc='Data Processing'):  # ncols: 进度条长度
            label, sentence = raw[-1], raw[1]  # 标签和文本
            sentence = Preprocessor.basic_pipeline(sentence)
            tmp = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer(sentence)]
            if len(tmp) > self.max_position_embeddings - 1:
                tmp = tmp[:self.max_position_embeddings - 1]  # BERT预训练模型只取前512个字符
            tmp += [self.SEP_IDX]
            tensor_ = torch.tensor(tmp, dtype=torch.long)  # 序列
            label = torch.tensor(int(label), dtype=torch.long)  # 标签
            max_len = max(max_len, tensor_.size(0))  # 用于保存最长序列的长度
            data.append((tensor_, label))
        return data, max_len

    def generate_batch(self, data_batch):
        """对每个batch的Token序列进行padding处理"""
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            batch_sentence.append(sen)
            batch_label.append(label)
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_label

    def load_train_val_test_data(self, train_file_path=None, val_file_path=None, test_file_path=None, only_test=False):
        postfix = str(self.max_sen_len)
        test_data, _ = self.data_process(filepath=test_file_path, postfix=postfix)
        test_iter = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.generate_batch)
        if only_test:
            return test_iter
        train_data, max_sen_len = self.data_process(filepath=train_file_path, postfix=postfix)  # 得到处理好的所有样本
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len  # 整个数据集中样本的最大长度
        val_data, _ = self.data_process(filepath=val_file_path, postfix=postfix)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch)
        return train_iter, test_iter, val_iter

    # BERT + document_topics
    @cache
    def data_process_with_document_topics(self, filepath, document_topics_file_path, postfix='cache'):
        topic_dists = load_document_topics(document_topics_file_path, recover_topic_peaks=False, max_m=None)
        raw_iter = pd.read_csv(filepath)
        data = []
        max_len = 0
        for ind, raw in tqdm(enumerate(raw_iter.values), desc='Data Processing'):
            label, sentence = raw[-1], raw[1]  # 标签和文本
            sentence = Preprocessor.basic_pipeline(sentence)
            tmp = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer(sentence)]
            if len(tmp) > self.max_position_embeddings - 1:
                tmp = tmp[:self.max_position_embeddings - 1]  # BERT预训练模型只取前512个字符
            tmp += [self.SEP_IDX]
            tensor_ = torch.tensor(tmp, dtype=torch.long)  # 序列
            label = torch.tensor(int(label), dtype=torch.long)  # 标签
            max_len = max(max_len, tensor_.size(0))  # 用于保存最长序列的长度

            topic_vector = topic_dists[ind]  # ndarray [sample, topic_nums]

            data.append((tensor_, label, topic_vector))
        return data, max_len

    def generate_batch_with_document_topics(self, data_batch):
        """对每个batch的Token序列进行padding处理"""
        batch_sentence, batch_label, batch_topic_vector = [], [], []
        for (sen, label, topic_vector) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            batch_sentence.append(sen)
            batch_label.append(label)
            batch_topic_vector.append(topic_vector)
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        batch_topic_vector = torch.tensor(np.array(batch_topic_vector), dtype=torch.float)
        return batch_sentence, batch_label, batch_topic_vector

    def load_data_with_document_topics(self, train_file_path=None, val_file_path=None, test_file_path=None,
                                       train_document_topics_file_path=None,
                                       val_document_topics_file_path=None,
                                       test_document_topics_file_path=None,
                                       only_test=False):
        postfix = str(self.max_sen_len)
        test_data, _ = self.data_process_with_document_topics(filepath=test_file_path,
                                                              document_topics_file_path=test_document_topics_file_path,
                                                              postfix=postfix)
        test_iter = DataLoader(test_data, batch_size=self.batch_size, shuffle=False,
                               collate_fn=self.generate_batch_with_document_topics)
        if only_test:
            return test_iter
        train_data, max_sen_len = self.data_process_with_document_topics(filepath=train_file_path,
                                                                         document_topics_file_path=train_document_topics_file_path,
                                                                         postfix=postfix)  # 得到处理好的所有样本
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len  # 整个数据集中样本的最大长度
        val_data, _ = self.data_process_with_document_topics(filepath=val_file_path,
                                                             document_topics_file_path=val_document_topics_file_path,
                                                             postfix=postfix)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch_with_document_topics)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch_with_document_topics)
        return train_iter, test_iter, val_iter

    # BERT + word topic
    @cache
    def data_process_with_word_topics(self, config, filepath, postfix='cache'):
        word_topics = load_word_topics(config, add_unk=True, recover_topic_peaks=False)
        word2id_dict = word_topics['word_id_dict']
        ltp = Preprocessor.ltp_init(config.user_dict_file_path)

        raw_iter = pd.read_csv(filepath)
        data = []
        max_len = 0
        for ind, raw in tqdm(enumerate(raw_iter.values), desc='Data Processing'):
            label, sentence_bert, sentence_topic = raw[-1], raw[1], raw[1]  # 标签和文本
            # 主题词映射
            # 数据预处理
            sentence_topic = Preprocessor.basic_pipeline(sentence_topic)
            sentence_topic = Preprocessor.process_for_segmented(sentence_topic)
            # 分词、映射
            segment, _ = ltp.seg([sentence_topic])
            word_ids = [word2id_dict[lemma] if lemma in word2id_dict.keys() else 0 for lemma in segment[0]]
            word_ids_tensor = torch.tensor(word_ids, dtype=torch.long)

            # bert词映射
            sentence_bert = Preprocessor.basic_pipeline(sentence_bert)
            tmp = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer(sentence_bert)]
            if len(tmp) > self.max_position_embeddings - 1:
                tmp = tmp[:self.max_position_embeddings - 1]  # BERT预训练模型只取前512个字符
            tmp += [self.SEP_IDX]
            tensor_ = torch.tensor(tmp, dtype=torch.long)  # 序列
            label = torch.tensor(int(label), dtype=torch.long)  # 标签
            max_len = max(max_len, tensor_.size(0))  # 用于保存最长序列的长度

            data.append((tensor_, label, word_ids_tensor))
        return data, max_len

    def generate_batch_with_word_topics(self, data_batch):
        """对每个batch的Token序列进行padding处理"""
        batch_sentence, batch_label, batch_word_topic, sen_topic_lengths = [], [], [], []
        for (sen_bert, label, sen_topic) in data_batch:  # 开始对一个batch中的每一个样本进行处理
            batch_sentence.append(sen_bert)
            batch_label.append(label)
            batch_word_topic.append(sen_topic)
            sen_topic_lengths.append(len(sen_topic))
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len)
        batch_word_topic = pad_sequence(batch_word_topic, batch_first=True, max_len=None, padding_value=0)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        sen_topic_lengths = torch.tensor(sen_topic_lengths, dtype=torch.float)
        return batch_sentence, batch_label, batch_word_topic, sen_topic_lengths

    def load_data_with_word_topics(self, config, train_file_path=None, val_file_path=None,
                                   test_file_path=None, only_test=False):
        postfix = str(self.max_sen_len)
        test_data, _ = self.data_process_with_word_topics(config, filepath=test_file_path, postfix=postfix)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch_with_word_topics)
        if only_test:
            return test_iter
        train_data, max_sen_len = self.data_process_with_word_topics(config, filepath=train_file_path, postfix=postfix)
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len  # 整个数据集中样本的最大长度
        val_data, _ = self.data_process_with_word_topics(config, filepath=val_file_path, postfix=postfix)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch_with_word_topics)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch_with_word_topics)
        return train_iter, test_iter, val_iter

    # BERT + document topic + weibo influence + user reliability
    @cache
    def data_process_with_document_and_ir(self, filepath, document_topics_file_path, manual_feature_file_path, postfix='cache'):
        # 加载文档-主题分布
        topic_dists = load_document_topics(document_topics_file_path, recover_topic_peaks=False, max_m=None)
        # 加载手工特征
        manual_feature = np.load(manual_feature_file_path)

        raw_iter = pd.read_csv(filepath)
        data = []
        max_len = 0
        for ind, raw in tqdm(enumerate(raw_iter.values), desc='Data Processing'):
            label, sentence = raw[-1], raw[1]  # 标签和文本
            sentence = Preprocessor.basic_pipeline(sentence)
            tmp = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer(sentence)]
            if len(tmp) > self.max_position_embeddings - 1:
                tmp = tmp[:self.max_position_embeddings - 1]  # BERT预训练模型只取前512个字符
            tmp += [self.SEP_IDX]
            tensor_ = torch.tensor(tmp, dtype=torch.long)  # 序列
            label = torch.tensor(int(label), dtype=torch.long)  # 标签
            max_len = max(max_len, tensor_.size(0))  # 用于保存最长序列的长度

            topic_vector = topic_dists[ind]  # ndarray [sample, topic_nums]
            feature_variable = manual_feature[ind]  # [weibo_influence, user_reliable]

            data.append((tensor_, label, topic_vector, feature_variable))
        return data, max_len

    def generate_batch_with_document_and_lr(self, data_batch):
        """对每个batch的Token序列进行padding处理"""
        batch_sentence, batch_label, batch_topic_vector, batch_manual_feature = [], [], [], []
        for (sen, label, topic_vector, feature_variable) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            batch_sentence.append(sen)
            batch_label.append(label)
            batch_topic_vector.append(topic_vector)
            batch_manual_feature.append(feature_variable)
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        batch_topic_vector = torch.tensor(np.array(batch_topic_vector), dtype=torch.float)
        batch_manual_feature = torch.tensor(np.array(batch_manual_feature), dtype=torch.float)
        return batch_sentence, batch_label, batch_topic_vector, batch_manual_feature

    def load_data_with_document_and_lr(self,
                                       train_file_path=None,
                                       val_file_path=None,
                                       test_file_path=None,
                                       train_document_topics_file_path=None,
                                       val_document_topics_file_path=None,
                                       test_document_topics_file_path=None,
                                       train_manual_feature_file_path=None,
                                       val_manual_feature_file_path=None,
                                       test_manual_feature_file_path=None,
                                       only_test=False):
        postfix = str(self.max_sen_len)
        test_data, _ = self.data_process_with_document_and_ir(filepath=test_file_path,
                                                              document_topics_file_path=test_document_topics_file_path,
                                                              manual_feature_file_path=test_manual_feature_file_path,
                                                              postfix=postfix)
        test_iter = DataLoader(test_data, batch_size=self.batch_size, shuffle=False,
                               collate_fn=self.generate_batch_with_document_and_lr)
        if only_test:
            return test_iter
        train_data, max_sen_len = self.data_process_with_document_and_ir(filepath=train_file_path,
                                                                         document_topics_file_path=train_document_topics_file_path,
                                                                         manual_feature_file_path=train_manual_feature_file_path,
                                                                         postfix=postfix)  # 得到处理好的所有样本
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len  # 整个数据集中样本的最大长度
        val_data, _ = self.data_process_with_document_and_ir(filepath=val_file_path,
                                                             document_topics_file_path=val_document_topics_file_path,
                                                             manual_feature_file_path=val_manual_feature_file_path,
                                                             postfix=postfix)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch_with_document_and_lr)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch_with_document_and_lr)
        return train_iter, test_iter, val_iter

    # BERT + word topic + weibo influence + user reliability
    @cache
    def data_process_with_word_and_ir(self, config, word_topics, filepath, manual_feature_file_path, postfix='cache'):
        word2id_dict = word_topics['word_id_dict']
        ltp = Preprocessor.ltp_init(config.user_dict_file_path)

        # 加载手工特征
        manual_feature = np.load(manual_feature_file_path)

        raw_iter = pd.read_csv(filepath)
        data = []
        max_len = 0
        for ind, raw in tqdm(enumerate(raw_iter.values), desc='Data Processing'):
            label, sentence_bert, sentence_topic = raw[-1], raw[1], raw[1]  # 标签和文本
            # 主题词映射
            # 数据预处理
            sentence_topic = Preprocessor.basic_pipeline(sentence_topic)
            sentence_topic = Preprocessor.process_for_segmented(sentence_topic)
            # 分词、映射
            segment, _ = ltp.seg([sentence_topic])
            word_ids = [word2id_dict[lemma] if lemma in word2id_dict.keys() else 0 for lemma in segment[0]]
            word_ids_tensor = torch.tensor(word_ids, dtype=torch.long)

            # bert词映射
            sentence_bert = Preprocessor.basic_pipeline(sentence_bert)
            tmp = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer(sentence_bert)]
            if len(tmp) > self.max_position_embeddings - 1:
                tmp = tmp[:self.max_position_embeddings - 1]  # BERT预训练模型只取前512个字符
            tmp += [self.SEP_IDX]
            tensor_ = torch.tensor(tmp, dtype=torch.long)  # 序列
            label = torch.tensor(int(label), dtype=torch.long)  # 标签
            max_len = max(max_len, tensor_.size(0))  # 用于保存最长序列的长度

            feature_variable = manual_feature[ind]  # [weibo_influence, user_reliable]

            data.append((tensor_, label, word_ids_tensor, feature_variable))
        return data, max_len

    def generate_batch_with_word_and_ir(self, data_batch):
        """对每个batch的Token序列进行padding处理"""
        batch_sentence, batch_label, batch_word_topic, batch_manual_feature = [], [], [], []
        for (sen_bert, label, sen_topic, feature_variable) in data_batch:  # 开始对一个batch中的每一个样本进行处理
            batch_sentence.append(sen_bert)
            batch_label.append(label)
            batch_word_topic.append(sen_topic)
            batch_manual_feature.append(feature_variable)
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len)
        batch_word_topic = pad_sequence(batch_word_topic, batch_first=True, max_len=None, padding_value=0)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        batch_manual_feature = torch.tensor(np.array(batch_manual_feature), dtype=torch.float)
        return batch_sentence, batch_label, batch_word_topic, batch_manual_feature

    def load_data_with_word_and_ir(self,
                                   config,
                                   word_topics=None,
                                   train_file_path=None,
                                   val_file_path=None,
                                   test_file_path=None,
                                   train_manual_feature_file_path=None,
                                   val_manual_feature_file_path=None,
                                   test_manual_feature_file_path=None,
                                   only_test=False):
        postfix = str(self.max_sen_len)
        test_data, _ = self.data_process_with_word_and_ir(config,
                                                          word_topics=word_topics,
                                                          filepath=test_file_path,
                                                          manual_feature_file_path=test_manual_feature_file_path,
                                                          postfix=postfix)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch_with_word_and_ir)
        if only_test:
            return test_iter
        train_data, max_sen_len = self.data_process_with_word_and_ir(config,
                                                                     word_topics=word_topics,
                                                                     filepath=train_file_path,
                                                                     manual_feature_file_path=train_manual_feature_file_path,
                                                                     postfix=postfix)
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len  # 整个数据集中样本的最大长度
        val_data, _ = self.data_process_with_word_and_ir(config,
                                                         word_topics=word_topics,
                                                         filepath=val_file_path,
                                                         manual_feature_file_path=val_manual_feature_file_path,
                                                         postfix=postfix)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=self.is_sample_shuffle, collate_fn=self.generate_batch_with_word_and_ir)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch_with_word_and_ir)
        return train_iter, test_iter, val_iter

    # BERT-W-I or BERT-W-R or BERT-I-R or W-I-R
    def data_process_for_ablation_study(self, config, data_filepath=None, segmented_file_path=None,
                                        word_topics=None, manual_feature_filepath=None, feature_name=None):
        word2id_dict = word_topics['word_id_dict']
        manual_feature = np.load(manual_feature_filepath)
        feature_influence = manual_feature[:, 0]  # weibo influence
        feature_reliability = manual_feature[:, 1]  # user reliability

        raw_iter = pd.read_csv(data_filepath)
        inputs_for_bert, labels, sentences_for_seg = [], [], []
        max_len = 0
        for raw in tqdm(raw_iter.values, desc='Data Processing'):
            label, sentence = raw[-1], raw[1]  # 标签和文本
            sentence = Preprocessor.basic_pipeline(sentence)
            # BERT
            tmp = [self.CLS_IDX] + [self.vocab[token] for token in self.tokenizer(sentence)]
            if len(tmp) > self.max_position_embeddings - 1:
                tmp = tmp[:self.max_position_embeddings - 1]  # BERT预训练模型只取前512个字符
            tmp += [self.SEP_IDX]
            tensor_ = torch.tensor(tmp, dtype=torch.long)  # 序列
            label = torch.tensor(int(label), dtype=torch.long)  # 标签
            max_len = max(max_len, tensor_.size(0))  # 用于保存最长序列的长度

            sentence_for_seg = Preprocessor.process_for_segmented(sentence)

            inputs_for_bert.append(tensor_)
            labels.append(label)
            sentences_for_seg.append(sentence_for_seg)

        data_tokenized = tokenize_ltp(sentences_for_seg, user_dict_filepath=config.user_dict_file_path,
                                      filepath=segmented_file_path, postfix=config.tokenize_type)
        # 句子级主题-词特征、微博影响力、用户可信度 特征提取
        word_topics_tensor, feature_influence_tensor, feature_reliability_tensor = [], [], []
        for ind, segment in enumerate(data_tokenized):
            word_ids = [word2id_dict[lemma] if lemma in word2id_dict.keys() else 0 for lemma in segment]
            word_ids_tensor = torch.tensor(word_ids, dtype=torch.long)
            # sentence_embedding = word_topics_embedding[word_ids, :].mean(axis=0)
            # sentence_embedding = torch.tensor(sentence_embedding, dtype=torch.float)
            word_topics_tensor.append(word_ids_tensor)

            influence = torch.tensor(feature_influence[ind], dtype=torch.float)
            reliability = torch.tensor(feature_reliability[ind], dtype=torch.float)
            feature_influence_tensor.append(influence)
            feature_reliability_tensor.append(reliability)

        if feature_name == 'BERT-W-I':
            data = []
            for ind in range(len(inputs_for_bert)):
                data.append((inputs_for_bert[ind], word_topics_tensor[ind],
                             feature_influence_tensor[ind], labels[ind]))

        elif feature_name == 'BERT-W-R':
            data = []
            for ind in range(len(inputs_for_bert)):
                data.append((inputs_for_bert[ind], word_topics_tensor[ind],
                             feature_reliability_tensor[ind], labels[ind]))

        elif feature_name == 'BERT-I-R':
            data = []
            for ind in range(len(inputs_for_bert)):
                data.append((inputs_for_bert[ind], feature_influence_tensor[ind],
                             feature_reliability_tensor[ind], labels[ind]))

        elif feature_name == 'W-I-R':
            data = []
            for ind in range(len(word_topics_tensor)):
                data.append((word_topics_tensor[ind], feature_influence_tensor[ind],
                             feature_reliability_tensor[ind], labels[ind]))
        else:
            raise TypeError

        data = DataMapping(data)
        return data, max_len

    def generate_batch_for_bert_w_i(self, data_batch):
        batch_sentences, batch_word_topics, batch_feature_influence, batch_labels = [], [], [], []
        for (sen_bert, sen_topic, feature_influence, label) in data_batch:  # 开始对一个batch中的每一个样本进行处理
            batch_sentences.append(sen_bert)
            batch_word_topics.append(sen_topic)
            batch_feature_influence.append(feature_influence)
            batch_labels.append(label)
        batch_sentences = pad_sequence(batch_sentences, batch_first=False,
                                       max_len=self.max_sen_len, padding_value=self.PAD_IDX)
        batch_word_topics = pad_sequence(batch_word_topics, batch_first=True, max_len=None, padding_value=0)
        # batch_word_topics = torch.tensor(batch_word_topics, dtype=torch.float)
        batch_feature_influence = torch.tensor(batch_feature_influence, dtype=torch.float).unsqueeze(1)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        return batch_sentences, batch_word_topics, batch_feature_influence, batch_labels

    def generate_batch_for_bert_w_r(self, data_batch):
        batch_sentences, batch_word_topics, batch_feature_reliability, batch_labels = [], [], [], []
        for (sen_bert, sen_topic, feature_reliability, label) in data_batch:
            batch_sentences.append(sen_bert)
            batch_word_topics.append(sen_topic)
            batch_feature_reliability.append(feature_reliability)
            batch_labels.append(label)
        batch_sentences = pad_sequence(batch_sentences, batch_first=False,
                                       max_len=self.max_sen_len, padding_value=self.PAD_IDX)
        batch_word_topics = pad_sequence(batch_word_topics, batch_first=True, max_len=None, padding_value=0)
        # batch_word_topics = torch.tensor(batch_word_topics, dtype=torch.float)
        batch_feature_reliability = torch.tensor(batch_feature_reliability, dtype=torch.float).unsqueeze(1)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        return batch_sentences, batch_word_topics, batch_feature_reliability, batch_labels

    def generate_batch_for_bert_i_r(self, data_batch):
        batch_sentences, batch_feature_influence, batch_feature_reliability, batch_labels = [], [], [], []
        for (sen_bert, feature_influence, feature_reliability, label) in data_batch:
            batch_sentences.append(sen_bert)
            batch_feature_influence.append(feature_influence)
            batch_feature_reliability.append(feature_reliability)
            batch_labels.append(label)
        batch_sentences = pad_sequence(batch_sentences, batch_first=False,
                                       max_len=self.max_sen_len, padding_value=self.PAD_IDX)
        batch_feature_influence = torch.tensor(batch_feature_influence, dtype=torch.float).unsqueeze(1)
        batch_feature_reliability = torch.tensor(batch_feature_reliability, dtype=torch.float).unsqueeze(1)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        return batch_sentences, batch_feature_influence, batch_feature_reliability, batch_labels

    def generate_batch_for_w_i_r(self, data_batch):
        batch_word_topics, batch_feature_influence, batch_feature_reliability, batch_labels = [], [], [], []
        for (sen_topic, feature_influence, feature_reliability, label) in data_batch:
            batch_word_topics.append(sen_topic)
            batch_feature_influence.append(feature_influence)
            batch_feature_reliability.append(feature_reliability)
            batch_labels.append(label)
        # batch_word_topics = torch.stack(batch_word_topics, dim=0)
        batch_word_topics = pad_sequence(batch_word_topics, batch_first=True, max_len=None, padding_value=0)
        batch_feature_influence = torch.tensor(batch_feature_influence, dtype=torch.float).unsqueeze(1)
        batch_feature_reliability = torch.tensor(batch_feature_reliability, dtype=torch.float).unsqueeze(1)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        return batch_word_topics, batch_feature_influence, batch_feature_reliability, batch_labels

    def load_data_for_ablation_study(self, config, only_test=False, feature_name=None, collate_fn=None):
        logging.info(f'##正在处理{feature_name}所需数据')
        word_topics = load_word_topics(config)  # 加载主题词表
        test_data, _ = self.data_process_for_ablation_study(config,
                                                            config.test_file_path,
                                                            config.segmented_test_file_path,
                                                            word_topics,
                                                            config.test_manual_features_file_path,
                                                            feature_name=feature_name)
        test_iter = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        if only_test:
            return test_iter
        train_data, _ = self.data_process_for_ablation_study(config,
                                                             config.train_file_path,
                                                             config.segmented_train_file_path,
                                                             word_topics,
                                                             config.train_manual_features_file_path,
                                                             feature_name=feature_name)
        val_data, _ = self.data_process_for_ablation_study(config,
                                                           config.val_file_path,
                                                           config.segmented_val_file_path,
                                                           word_topics,
                                                           config.val_manual_features_file_path,
                                                           feature_name=feature_name)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=self.is_sample_shuffle, collate_fn=collate_fn)
        val_iter = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        return train_iter, val_iter


class DataMapping(Dataset):
    """数据映射"""
    def __init__(self, data):
        self.dataset = data
        self.lens = len(data)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)




if __name__ == '__main__':
    pass
    from transformers import BertTokenizer
    # filepath = '../data/test.csv'
    # raw_iter = pd.read_csv(filepath)
    # pretrained_model_dir = r"D:\Desktop\BERT\jupyter\BertTextClassification\bert_base_chinese"
    # vocab = build_vocab(r"D:\Desktop\BERT\jupyter\BertTextClassification\bert_base_chinese\vocab.txt")
    # bert_tokenize = BertTokenizer.from_pretrained(pretrained_model_dir).tokenize
    # for raw in tqdm(raw_iter.values[:7], desc='Data Processing', ncols=80):  # ncols: 进度条长度
    #     l, s = raw[-1], raw[1]  # 标签和文本
    #     s = Preprocessor.basic_pipeline(s)
    #     # tmp = [token for token in bert_tokenize(s)]
    #     tmp = [vocab['[CLS]']] + [vocab[token] for token in bert_tokenize(s)]
    #     print(tmp)