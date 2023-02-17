# 此模块进行数据的处理、封装等，将会完成模型训练前的所有数据操作
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from baseline.utils.data_preprocessor import Preprocessor, tokenize_ltp
from baseline.model.word2vec import load_word2vec_vector
from topic_model_sklearn.topic_loader import load_document_topics


class DataMapping(Dataset):
    """数据映射"""
    def __init__(self, data):
        self.dataset = data
        self.lens = len(data)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class CollateFn:
    """迭代训练前的数据整理"""
    @staticmethod
    def generate_batch_textcnn(examples):
        """对一个批次内的数据进行处理"""
        inputs = [torch.tensor(ex[0], dtype=torch.long) for ex in examples]
        labels = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        # 对批次内的样本进行补齐，使其具有相同长度
        inputs = pad_sequence(inputs, batch_first=True)
        return inputs, labels

    @staticmethod
    def generate_batch_lstm(examples):
        lengths = torch.tensor([len(ex[0]) for ex in examples])
        inputs = [torch.tensor(ex[0], dtype=torch.long) for ex in examples]
        labels = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        inputs = pad_sequence(inputs, batch_first=True)
        return inputs, lengths, labels

    @staticmethod
    def generate_batch_word2vec(examples):
        word_vector = torch.tensor(np.array([ex[0] for ex in examples]), dtype=torch.float)
        topic_vector = torch.tensor(np.array([ex[1] for ex in examples]), dtype=torch.float)
        labels = torch.tensor([ex[2] for ex in examples], dtype=torch.long)
        return word_vector, topic_vector, labels

    @staticmethod
    def generate_batch_logistic_regression(examples):
        inputs = torch.tensor(np.array([ex[0] for ex in examples]), dtype=torch.float)
        labels = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        return inputs, labels


class FakeNewsDataset:
    def __init__(self, config, model_name=None):
        # 数据集
        self.train_data_filepath = config.train_data_filepath
        self.val_data_filepath = config.val_data_filepath
        self.test_data_filepath = config.test_data_filepath
        self.user_dict_filepath = config.user_dict_filepath
        # 分词缓存
        self.segmented_train_filepath = config.segmented_train_filepath
        self.segmented_val_filepath = config.segmented_val_filepath
        self.segmented_test_filepath = config.segmented_test_filepath

        self.tokenize = config.tokenize
        self.tokenize_type = config.tokenize_type
        self.batch_size = config.batch_size

        if model_name == 'TextCNN' or model_name == 'Bi-LSTM':
            self.vocab = Preprocessor.build_vocab(config, tokenize=self.tokenize)
            self.vocab_size = len(self.vocab)

        elif model_name == 'Word2vecLDA':
            # word2vec + LDA
            self.word_vector = load_word2vec_vector(config, add_unk=True)
            self.word_embedding = self.word_vector['vector_matrix']  # 词向量表
            self.word2id_dict = self.word_vector['word2id_dict']
            self.train_topic_dists = load_document_topics(config.train_document_topics_filepath)
            self.val_topic_dists = load_document_topics(config.val_document_topics_filepath)
            self.test_topic_dists = load_document_topics(config.test_document_topics_filepath)

        elif model_name == 'LogisticRegression':
            # LDA + I + R
            self.train_topic_dists = load_document_topics(config.train_document_topics_filepath)
            self.val_topic_dists = load_document_topics(config.val_document_topics_filepath)
            self.test_topic_dists = load_document_topics(config.test_document_topics_filepath)
            self.train_manual_features = np.load(config.train_manual_features_filepath)
            self.val_manual_features = np.load(config.val_manual_features_filepath)
            self.test_manual_features = np.load(config.test_manual_features_filepath)
        else:
            pass

    def data_process(self, data_filepath, segmented_filepath):
        """数据处理"""
        raw_iter = pd.read_csv(data_filepath)
        labels, sentences = [], []
        for ind, raw in tqdm(enumerate(raw_iter.values), desc='Data Processing'):
            label, s = raw[-1], raw[1]  # 标签和文本
            s = Preprocessor.basic_pipeline(s)
            s = Preprocessor.process_for_segmented(s)
            sentences.append(s)
            labels.append(label)
        # 分词
        sentence_tokenized = tokenize_ltp(sentences,
                                          user_dict_filepath=self.user_dict_filepath,
                                          filepath=segmented_filepath,
                                          postfix=self.tokenize_type)
        # sentence_tokenized -> list
        # sentence_tokenized[0] -> list
        # sentence_tokenized[0][0] -> str token

        data = [(self.vocab.convert_tokens_to_ids(segment), labels[ind]) for ind, segment in
                enumerate(sentence_tokenized)]

        data = DataMapping(data)
        return data

    def load_data(self, only_test=False, collate_fn=None, model_name=None):
        """封装数据用于迭代训练"""
        if model_name == 'TextCNN' or model_name == 'Bi-LSTM':
            print(f'正在处理{model_name}所需数据')
            test_data = self.data_process(self.test_data_filepath, self.segmented_test_filepath)
            test_iter = DataLoader(test_data, batch_size=self.batch_size, shuffle=False,
                                   collate_fn=collate_fn)
            if only_test:
                return test_iter

            train_data = self.data_process(self.train_data_filepath, self.segmented_train_filepath)
            val_data = self.data_process(self.val_data_filepath, self.segmented_val_filepath)

        elif model_name == 'Word2vecLDA':
            print(f'正在处理{model_name}所需数据')
            test_data = self.data_process_for_word2vec(self.test_data_filepath,
                                                       self.segmented_test_filepath,
                                                       self.test_topic_dists)
            test_iter = DataLoader(test_data, batch_size=self.batch_size, shuffle=False,
                                   collate_fn=collate_fn)
            if only_test:
                return test_iter
            train_data = self.data_process_for_word2vec(self.train_data_filepath,
                                                        self.segmented_train_filepath,
                                                        self.train_topic_dists)
            val_data = self.data_process_for_word2vec(self.val_data_filepath,
                                                      self.segmented_val_filepath,
                                                      self.val_topic_dists)

        elif model_name == 'LogisticRegression':
            print(f'正在处理{model_name}所需数据')
            test_data = self.data_process_for_logistic_regression(self.test_data_filepath,
                                                                  self.test_topic_dists,
                                                                  self.test_manual_features)
            test_iter = DataLoader(test_data, batch_size=self.batch_size, shuffle=False,
                                   collate_fn=collate_fn)
            if only_test:
                return test_iter
            train_data = self.data_process_for_logistic_regression(self.train_data_filepath,
                                                                   self.train_topic_dists,
                                                                   self.train_manual_features)
            val_data = self.data_process_for_logistic_regression(self.val_data_filepath,
                                                                 self.val_topic_dists,
                                                                 self.val_manual_features)
        else:
            print('模型名称输入有误。')
            return

        train_iter = DataLoader(train_data, batch_size=self.batch_size, shuffle=True,
                                collate_fn=collate_fn)
        val_iter = DataLoader(val_data, batch_size=self.batch_size, shuffle=False,
                              collate_fn=collate_fn)

        return train_iter, val_iter

    def data_process_for_word2vec(self, data_filepath, segmented_filepath, topic_dists):
        """
        :param data_filepath: 数据集位置
        :param segmented_filepath: 分词缓存文件位置
        :param topic_dists: 文档-主题分布
        """
        raw_iter = pd.read_csv(data_filepath)
        labels, sentences = [], []
        for ind, raw in tqdm(enumerate(raw_iter.values), desc='Data Processing'):
            label, s = raw[-1], raw[1]  # 标签和文本
            s = Preprocessor.basic_pipeline(s)
            s = Preprocessor.process_for_segmented(s)
            sentences.append(s)
            labels.append(label)
        # 分词
        sentence_tokenized = tokenize_ltp(sentences,
                                          user_dict_filepath=self.user_dict_filepath,
                                          filepath=segmented_filepath,
                                          postfix=self.tokenize_type)
        data = []
        for ind, segment in enumerate(sentence_tokenized):
            # 词向量提取
            word_ids = [self.word2id_dict[lemma] if lemma in self.word2id_dict.keys() else 0 for lemma in segment]
            sentence_embedding = self.word_embedding[word_ids, :].mean(axis=0)
            # 主题向量提取
            topic_vector = topic_dists[ind]

            data.append((sentence_embedding, topic_vector, labels[ind]))

        data = DataMapping(data)
        return data

    def data_process_for_logistic_regression(self, data_filepath, topic_dists, manual_features):
        features = np.concatenate((topic_dists, manual_features), axis=1)  # 特征合并
        raw_iter = pd.read_csv(data_filepath)
        data = []
        for ind, raw in enumerate(raw_iter.values):
            label = raw[-1]  # 标签
            feature = features[ind]
            data.append((feature, label))
        data = DataMapping(data)
        return data


if __name__ == '__main__':
    from baseline.baseline_config import Config
    config = Config()
    dataset = FakeNewsDataset(config)
    # data_ = dataset.data_process_for_word2vec(config.train_data_filepath,
    #                                           config.segmented_train_filepath, dataset.train_topic_dists)
    # train_iter, val_iter = dataset.load_data(only_test=False, collate_fn=CollateFn.generate_batch_word2vec,
    #                                          model_name='Word2vecLDA')




























