import os
import time
import logging
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from topic_config import TopicConfig
from topic_trainer import lda_preprocess
from topic_predictor import infer_and_write_document_topics, infer_and_write_word_topics
from topic_trainer import train_topic_model
from topic_loader import load_document_topics, load_word_topics
from utils.data_helpers import Preprocessor, tokenize_ltp, pad_sequence


def train_lda(topic_config):
    train_file_path = topic_config.train_file_path
    raw_iter = pd.read_csv(train_file_path)
    sentences = []
    for raw in tqdm(raw_iter.values, desc='Data Processing'):
        sentence = raw[1]  # 文本
        sentence_1 = Preprocessor.basic_pipeline(sentence)  # 替换文本中的url、@id、繁体转简体
        sentence_2 = Preprocessor.process_for_segmented(sentence_1)
        sentences.append(sentence_2)

    # ltp分词
    data_tokenized = tokenize_ltp(sentences, user_dict_filepath=topic_config.user_dict_file_path,
                                  filepath=topic_config.segmented_train_file_path, postfix=topic_config.tokenize_type)
    # jieba分词
    # data_tokenized = tokenize_jieba(sentences, user_dict_filepath=topic_config.user_dict_file_path,
    #                                 filepath=topic_config.segmented_file_path, postfix=topic_config.tokenize_type)

    # 数据处理
    corpus, id2word, processed_texts = lda_preprocess(data_tokenized, id2word=None, delete_stopwords=True,
                                                      stopwords_file_path=topic_config.stopwords_file_path,
                                                      processed_text_file_path=topic_config.processed_file_path,
                                                      print_steps=False)
    # 模型训练
    topic_model = train_topic_model(corpus, topic_config, save_model=True)

    # 推断文档-主题分布
    infer_and_write_document_topics(topic_config, topic_model=topic_model, id2word=id2word)
    # 推断主题-词分布
    infer_and_write_word_topics(topic_config, topic_model=topic_model, id2word=id2word, max_vocab=None)


class DataMapping(Dataset):
    """数据映射"""
    def __init__(self, data):
        self.dataset = data
        self.lens = len(data)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class TopicModelDataset:
    def __init__(self, topic_config):
        self.topic_config = topic_config
        self.batch_size = topic_config.batch_size
        # 数据集
        self.train_filepath = topic_config.train_file_path
        self.val_filepath = topic_config.val_file_path
        self.test_filepath = topic_config.test_file_path
        # 句子级主题特征
        self.train_document_topics_filepath = topic_config.train_document_topics_file_path
        self.val_document_topics_filepath = topic_config.val_document_topics_file_path
        self.test_document_topics_filepath = topic_config.test_document_topics_file_path
        # 词级别主题特征
        self.word_topics = load_word_topics(topic_config, add_unk=True, recover_topic_peaks=False)
        self.word2id_dict = self.word_topics['word_id_dict']
        self.word_topics_embedding = torch.tensor(self.word_topics['topic_matrix'],
                                                  dtype=torch.float, device=topic_config.device)
        # 分词缓存文件
        self.segmented_train_filepath = topic_config.segmented_train_file_path
        self.segmented_val_filepath = topic_config.segmented_val_file_path
        self.segmented_test_filepath = topic_config.segmented_test_file_path
        # 手工特征  微博影响力和用户可信度
        self.train_manual_features_filepath = topic_config.train_manual_features_file_path
        self.val_manual_features_filepath = topic_config.val_manual_features_file_path
        self.test_manual_features_filepath = topic_config.test_manual_features_file_path

    def data_process_for_topic_model(self, data_filepath, document_topics_filepath=None, segmented_filepath=None,
                                     manual_feature_filepath=None, topics_type=None, feature_name=None):
        """
        :param data_filepath: 数据集路径
        :param document_topics_filepath: 文档主题分布路径
        :param segmented_filepath: 分词文件缓存路径
        :param manual_feature_filepath: 手工特征路径
        :param topics_type: 主题特征类型 句子级主题特征 or 词级别主题特征
        :param feature_name: 手工特征名称 改进后的微博影响力 or 原来的微博影响力
        """
        raw_iter = pd.read_csv(data_filepath)
        labels = [raw[-1] for raw in raw_iter.values]  # 标签
        if topics_type == 'DocumentTopics':
            topic_dists = load_document_topics(document_topics_filepath, recover_topic_peaks=False, max_m=None)
            if feature_name == 'D-NewInfluence' or feature_name == 'D-OldInfluence':
                manual_feature = np.load(manual_feature_filepath)
                if feature_name == 'D-NewInfluence':
                    feature_influence = manual_feature[:, 0]  # new weibo influence
                elif feature_name == 'D-OldInfluence':
                    feature_influence = manual_feature[:, 2]  # old weibo influence
                else:
                    raise TypeError

                data = []
                for ind in range(len(labels)):
                    label = torch.tensor(labels[ind], dtype=torch.long)
                    topic_vector = torch.tensor(topic_dists[ind], dtype=torch.float)
                    influence = torch.tensor(feature_influence[ind], dtype=torch.float)
                    data.append((topic_vector, influence, label))
                data = DataMapping(data)
                return data

            elif feature_name == 'only_document_topics':   # 只用句子级主题特征
                data = []
                for ind in range(len(labels)):
                    label = torch.tensor(labels[ind], dtype=torch.long)
                    topic_vector = torch.tensor(topic_dists[ind], dtype=torch.float)
                    data.append((topic_vector, label))
                data = DataMapping(data)
                return data

        elif topics_type == 'WordTopics':
            if feature_name == 'only_word_topics':
                sentences = []
                for raw in tqdm(raw_iter.values, desc='Data Processing'):
                    sentence = raw[1]  # 文本
                    sentence_1 = Preprocessor.basic_pipeline(sentence)  # 替换文本中的url、@id、繁体转简体
                    sentence_2 = Preprocessor.process_for_segmented(sentence_1)
                    sentences.append(sentence_2)

                data_tokenized = tokenize_ltp(sentences, user_dict_filepath=self.topic_config.user_dict_file_path,
                                              filepath=segmented_filepath, postfix=self.topic_config.tokenize_type)
                data = []
                for ind, segment in enumerate(data_tokenized):
                    word_ids = [self.word2id_dict[lemma] if lemma in self.word2id_dict.keys() else 0 for lemma in segment]
                    word_ids_tensor = torch.tensor(word_ids, dtype=torch.long)
                    label = torch.tensor(labels[ind], dtype=torch.long)
                    data.append((word_ids_tensor, label))
                data = DataMapping(data)
                return data

            else:
                raise TypeError

        else:
            raise TypeError

    def load_data_for_topic_model(self, only_test=False, topics_type=None, feature_name=None, collate_fn=None):
        logging.info(f'##正在处理{topics_type}所需数据')
        if feature_name is not None:
            logging.info(f'##正在处理{feature_name}所需数据')
        test_data = self.data_process_for_topic_model(self.test_filepath,
                                                      self.test_document_topics_filepath,
                                                      self.segmented_test_filepath,
                                                      self.test_manual_features_filepath,
                                                      topics_type=topics_type,
                                                      feature_name=feature_name)
        test_iter = DataLoader(test_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        if only_test:
            return test_iter
        train_data = self.data_process_for_topic_model(self.train_filepath,
                                                       self.train_document_topics_filepath,
                                                       self.segmented_train_filepath,
                                                       self.train_manual_features_filepath,
                                                       topics_type=topics_type,
                                                       feature_name=feature_name)
        val_data = self.data_process_for_topic_model(self.val_filepath,
                                                     self.val_document_topics_filepath,
                                                     self.segmented_val_filepath,
                                                     self.val_manual_features_filepath,
                                                     topics_type=topics_type,
                                                     feature_name=feature_name)
        train_iter = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        val_iter = DataLoader(val_data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        return train_iter, val_iter


class CollateFn:
    """批次内数据整理"""
    @staticmethod
    def generate_batch_for_document_topics(data_batch):
        """句子级主题特征"""
        batch_vector, batch_labels = [], []
        for (topic_vector, label) in data_batch:
            batch_vector.append(topic_vector)
            batch_labels.append(label)

        batch_vector = torch.stack(batch_vector)  # [batch_size, num_topics]
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        return batch_vector, batch_labels

    @staticmethod
    def generate_batch_for_word_topics(data_batch):
        """词级别主题特征"""
        batch_word_ids, batch_labels = [], []
        for (word_ids, label) in data_batch:
            batch_word_ids.append(word_ids)
            batch_labels.append(label)

        batch_word_ids = pad_sequence(batch_word_ids, batch_first=True, max_len=None, padding_value=0)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        return batch_word_ids, batch_labels

    @staticmethod
    def generate_batch_for_d_i(data_batch):
        """句子级主题特征+微博影响力"""
        batch_vector, batch_feature_influence, batch_labels = [], [], []
        for (topic_vector, feature_influence, label) in data_batch:
            batch_vector.append(topic_vector)
            batch_feature_influence.append(feature_influence)
            batch_labels.append(label)

        batch_vector = torch.stack(batch_vector)
        batch_feature_influence = torch.tensor(batch_feature_influence, dtype=torch.float).unsqueeze(1)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        return batch_vector, batch_feature_influence, batch_labels


class TopicModel(nn.Module):
    def __init__(self, topic_config, feature_name=None):
        super(TopicModel, self).__init__()
        if feature_name == 'D-NewInfluence' or feature_name == 'D-OldInfluence':
            self.hidden_dim = topic_config.num_topics + 1
        else:
            self.hidden_dim = topic_config.num_topics

        self.num_labels = topic_config.num_labels
        self.full_connection_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.normalize = nn.LayerNorm(self.hidden_dim)
        self.activate = torch.relu
        self.classifier = nn.Linear(self.hidden_dim, self.num_labels)

    def forward(self, document_topic_vector=None, word_topics_id=None,
                word_topics_embedding=None, feature_influence=None, labels=None):
        input_features = None
        if document_topic_vector is not None:
            input_features = document_topic_vector
            if feature_influence is not None:
                input_features = torch.cat([input_features, feature_influence], dim=1)

        elif word_topics_id is not None:
            word_topics_tensor = [torch.index_select(word_topics_embedding, dim=0, index=word_topics_id[ind]) for ind in
                                  range(word_topics_id.shape[0])]
            word_topics_tensor = torch.stack(word_topics_tensor)  # [batch_size, max_src_len, topic_nums]
            topic_vector = torch.mean(word_topics_tensor, dim=1)
            input_features = topic_vector

        fc_layer_out = self.full_connection_layer(input_features)
        ln_layer = self.normalize(fc_layer_out)
        pooled_out = self.activate(ln_layer)
        logits = self.classifier(pooled_out)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits

        else:
            return logits


def train(topic_config, topics_type=None, feature_name=None):
    model = TopicModel(topic_config, feature_name=feature_name)
    model_save_path = os.path.join(topic_config.model_save_dir, f'{feature_name}.pt')
    model = model.to(topic_config.device)

    lr_lambda = lambda epoch: 0.90 ** epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=topic_config.learning_rate)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)  # 创建学习率调度器
    model.train()

    data_loader = TopicModelDataset(topic_config)
    if topics_type == 'DocumentTopics':
        if feature_name == 'D-NewInfluence':
            train_iter, val_iter = \
                data_loader.load_data_for_topic_model(only_test=False, topics_type=topics_type, feature_name=feature_name,
                                                      collate_fn=CollateFn.generate_batch_for_d_i)
        elif feature_name == 'D-OldInfluence':
            train_iter, val_iter = \
                data_loader.load_data_for_topic_model(only_test=False, topics_type=topics_type, feature_name=feature_name,
                                                      collate_fn=CollateFn.generate_batch_for_d_i)
        elif feature_name == 'only_document_topics':
            train_iter, val_iter =\
                data_loader.load_data_for_topic_model(only_test=False, topics_type=topics_type, feature_name=feature_name,
                                                      collate_fn=CollateFn.generate_batch_for_document_topics)
        else:
            raise TypeError

    elif topics_type == 'WordTopics':
        if feature_name == 'only_word_topics':
            train_iter, val_iter = \
                data_loader.load_data_for_topic_model(only_test=False, topics_type=topics_type, feature_name=feature_name,
                                                      collate_fn=CollateFn.generate_batch_for_word_topics)
        else:
            raise TypeError

    else:
        raise TypeError

    max_f = 0
    for epoch in range(topic_config.epochs):
        losses = 0
        start_time = time.time()
        for idx, batch in enumerate(train_iter):
            if topics_type == 'DocumentTopics':
                if feature_name == 'D-NewInfluence' or feature_name == 'D-OldInfluence':
                    topic_vector, feature_influence, labels = batch
                    topic_vector = topic_vector.to(topic_config.device)
                    feature_influence = feature_influence.to(topic_config.device)
                    labels = labels.to(topic_config.device)

                    loss, logits = model(document_topic_vector=topic_vector,
                                         feature_influence=feature_influence,
                                         labels=labels)
                else:
                    topic_vector, labels = batch
                    topic_vector = topic_vector.to(topic_config.device)
                    labels = labels.to(topic_config.device)

                    loss, logits = model(document_topic_vector=topic_vector, labels=labels)

            elif topics_type == 'WordTopics':
                word_sample, labels = batch
                word_sample = word_sample.to(topic_config.device)
                labels = labels.to(topic_config.device)

                loss, logits = model(word_topics_id=word_sample,
                                     word_topics_embedding=data_loader.word_topics_embedding,
                                     labels=labels)
            else:
                raise TypeError

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            # acc = (logits.argmax(1) == labels).float().mean()
        scheduler.step()
        end_time = time.time()
        train_loss = losses
        logging.info(f"Epoch: {epoch + 1}, Train loss: {train_loss:.4f}, Epoch time = {(end_time - start_time):.4f}s")
        acc, p, r, f, _, _ = evaluate(val_iter, model, topic_config.device, topics_type=topics_type,
                                      feature_name=feature_name, word_topics_embedding=data_loader.word_topics_embedding)
        logging.info(f"Val: Epoch {epoch + 1} Accuracy: {acc:.4f},  Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")
        if f > max_f:
            max_f = f
            torch.save(model.state_dict(), model_save_path)


def evaluate(data_iter, model, device, topics_type=None, feature_name=None, word_topics_embedding=None):
    model.eval()
    with torch.no_grad():
        y_true, y_pred = [], []
        test_losses = 0
        # 发现一件特别神奇的事情，加与不加tqdm会影响最终预测效果
        # for batch in data_iter:
        for batch in tqdm(data_iter, desc='Evaluating'):
            if topics_type == 'DocumentTopics':
                if feature_name == 'D-NewInfluence' or feature_name == 'D-OldInfluence':
                    topic_vector, feature_influence, labels = batch
                    topic_vector = topic_vector.to(device)
                    feature_influence = feature_influence.to(device)
                    labels = labels.to(device)

                    loss, logits = model(document_topic_vector=topic_vector,
                                         feature_influence=feature_influence,
                                         labels=labels)
                else:
                    topic_vector, labels = batch
                    topic_vector = topic_vector.to(device)
                    labels = labels.to(device)

                    loss, logits = model(document_topic_vector=topic_vector, labels=labels)

            elif topics_type == 'WordTopics':
                word_sample, labels = batch
                word_sample = word_sample.to(device)
                labels = labels.to(device)

                loss, logits = model(word_topics_id=word_sample,
                                     word_topics_embedding=word_topics_embedding,
                                     labels=labels)
            else:
                raise TypeError

            test_losses += loss.item()
            batch_pred = logits.argmax(dim=1).tolist()  # 得到一个batch的预测标签
            batch_true = labels.tolist()
            y_pred.extend(batch_pred)
            y_true.extend(batch_true)

        logging.info(f'Test Loss: {test_losses:.4f}')
        acc = accuracy_score(y_true, y_pred)
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f = f1_score(y_true, y_pred)

    model.train()
    return acc, p, r, f, y_true, y_pred


def inference(topic_config, topics_type=None, feature_name=None):
    model = TopicModel(topic_config, feature_name=feature_name)
    model_save_path = os.path.join(topic_config.model_save_dir, f'{feature_name}.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行预测......")
    model = model.to(topic_config.device)

    data_loader = TopicModelDataset(topic_config)
    if topics_type == 'DocumentTopics':
        if feature_name == 'D-NewInfluence':
            test_iter = \
                data_loader.load_data_for_topic_model(only_test=True, topics_type=topics_type, feature_name=feature_name,
                                                      collate_fn=CollateFn.generate_batch_for_d_i)
        elif feature_name == 'D-OldInfluence':
            test_iter = \
                data_loader.load_data_for_topic_model(only_test=True, topics_type=topics_type, feature_name=feature_name,
                                                      collate_fn=CollateFn.generate_batch_for_d_i)
        elif feature_name == 'only_document_topics':
            test_iter = \
                data_loader.load_data_for_topic_model(only_test=True, topics_type=topics_type, feature_name=feature_name,
                                                      collate_fn=CollateFn.generate_batch_for_document_topics)
        else:
            raise TypeError

    elif topics_type == 'WordTopics':
        if feature_name == 'only_word_topics':
            test_iter = \
                data_loader.load_data_for_topic_model(only_test=True, topics_type=topics_type, feature_name=feature_name,
                                                      collate_fn=CollateFn.generate_batch_for_word_topics)
        else:
            raise TypeError

    else:
        raise TypeError

    acc, p, r, f, _, _ = evaluate(test_iter, model, topic_config.device, topics_type=topics_type,
                                  feature_name=feature_name, word_topics_embedding=data_loader.word_topics_embedding)
    logging.info(f"Test Results:    Accuracy: {acc:.4f},  Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")


if __name__ == '__main__':
    torch.manual_seed(1234)
    config = TopicConfig()
    # 训练lda模型
    # train_lda(config)

    # topics_type: 'DocumentTopics' or 'WordTopics'
    # feature_name: 'only_document_topics' or 'D-NewInfluence' or 'D-OldInfluence' or 'only_word_topics'

    # train(config, topics_type='DocumentTopics', feature_name='D-NewInfluence')
    # inference(config, topics_type='DocumentTopics', feature_name='D-NewInfluence')

    train(config, topics_type='DocumentTopics', feature_name='D-OldInfluence')
    inference(config, topics_type='DocumentTopics', feature_name='D-OldInfluence')





























