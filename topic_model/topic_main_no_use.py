import os
import time
import logging
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


# document topic
def data_process_with_document_topics(filepath, document_topics_file_path):

    topic_dists = load_document_topics(document_topics_file_path, recover_topic_peaks=False, max_m=None)

    raw_iter = pd.read_csv(filepath)
    features, labels = [], []
    for ind, raw in enumerate(raw_iter.values):
        label, sentence = raw[-1], raw[1]  # 标签和文本
        label = torch.tensor(int(label), dtype=torch.long)  # 标签
        topic_vector = torch.tensor(topic_dists[ind], dtype=torch.float)  # ndarray [sample, topic_nums]
        features.append(topic_vector)
        labels.append(label)
        # data.append((topic_vector, label))
    data = (features, labels)
    return data


def generate_batch_with_document_topics(data_batch):
    batch_vector, batch_label = [], []
    for (topic_vector, label) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
        # topic_vector = torch.tensor(topic_vector, dtype=torch.float)
        batch_vector.append(topic_vector)
        batch_label.append(label)

    batch_vector = torch.stack(batch_vector)
    batch_label = torch.tensor(batch_label, dtype=torch.long)
    return batch_vector, batch_label


# word topic
def data_process_with_word_topics(topic_config, data_file_path, segmented_file_path, word_topics):
    word2id_dict = word_topics['word_id_dict']
    raw_iter = pd.read_csv(data_file_path)
    labels, sentences = [], []
    for raw in tqdm(raw_iter.values, desc='Data Processing'):
        label, sentence = raw[-1], raw[1]  # 文本
        sentence_1 = Preprocessor.basic_pipeline(sentence)  # 替换文本中的url、@id、繁体转简体
        sentence_2 = Preprocessor.process_for_segmented(sentence_1)
        sentences.append(sentence_2)
        labels.append(label)

    # ltp分词
    data_tokenized = tokenize_ltp(sentences, user_dict_filepath=topic_config.user_dict_file_path,
                                  filepath=segmented_file_path, postfix=topic_config.tokenize_type)

    word_topics_id = []
    for segment in data_tokenized:
        word_ids = [word2id_dict[lemma] if lemma in word2id_dict.keys() else 0 for lemma in segment]
        word_ids_tensor = torch.tensor(word_ids, dtype=torch.long)
        word_topics_id.append(word_ids_tensor)
    data = (word_topics_id, labels)
    return data


def generate_batch_with_word_topics(data_batch):
    batch_word_topic, batch_label = [], []
    for word_ids, label in data_batch:
        batch_word_topic.append(word_ids)
        batch_label.append(label)
    # 批次内长度补齐
    batch_word_topic = pad_sequence(batch_word_topic, batch_first=True, max_len=None, padding_value=0)
    batch_label = torch.tensor(batch_label, dtype=torch.long)
    return batch_word_topic, batch_label


class TopicDataset(Dataset):
    def __init__(self, data):
        self.features = data[0]
        self.labels = data[1]
        assert len(data[0]) == len(data[1])
        self.lens = len(data[0])

    def __getitem__(self, item):
        return self.features[item], self.labels[item]

    def __len__(self):
        return self.lens


class DocumentTopicModel(nn.Module):
    def __init__(self, input_dim=20, num_labels=2):
        super(DocumentTopicModel, self).__init__()
        self.full_connection_layer = nn.Linear(input_dim, input_dim)
        self.classifier = nn.Linear(input_dim, num_labels)
        self.normalize = nn.LayerNorm(input_dim)
        self.activate = torch.tanh

    def forward(self, topic_vector):
        fc_layer_out = self.full_connection_layer(topic_vector)
        ln_layer = self.normalize(fc_layer_out)
        pooled_out = self.activate(ln_layer)
        outputs = self.classifier(pooled_out)
        return outputs


class WordTopicModel(nn.Module):
    def __init__(self, input_dim=100, num_labels=2):
        super(WordTopicModel, self).__init__()
        self.full_connection_layer = nn.Linear(input_dim, input_dim)
        self.classifier = nn.Linear(input_dim, num_labels)
        self.normalize = nn.LayerNorm(input_dim)
        self.activate = torch.tanh

    def forward(self, word_topic_ids, topic_matrix):
        word_topics_tensor = [torch.index_select(topic_matrix, dim=0, index=word_topic_ids[ind]) for ind in
                              range(word_topic_ids.shape[0])]
        word_topics_tensor = torch.stack(word_topics_tensor)  # [batch_size, max_src_len, topic_nums]
        topic_vector = torch.mean(word_topics_tensor, dim=1)

        fc_layer_out = self.full_connection_layer(topic_vector)
        ln_layer = self.normalize(fc_layer_out)
        pooled_out = self.activate(ln_layer)
        outputs = self.classifier(pooled_out)
        return outputs


def train_document_topic(topic_config):
    train_data = data_process_with_document_topics(topic_config.train_file_path,
                                                   topic_config.train_document_topics_file_path)
    test_data = data_process_with_document_topics(topic_config.test_file_path,
                                                  topic_config.test_document_topics_file_path)
    train_dataset = TopicDataset(train_data)
    test_dataset = TopicDataset(test_data)
    train_iter = DataLoader(train_dataset, batch_size=topic_config.batch_size, shuffle=True,
                            collate_fn=generate_batch_with_document_topics)
    test_iter = DataLoader(test_dataset, batch_size=topic_config.batch_size, shuffle=False,
                           collate_fn=generate_batch_with_document_topics)

    model = DocumentTopicModel(input_dim=topic_config.num_topics, num_labels=topic_config.num_labels)
    model_save_path = os.path.join(topic_config.topic_results_save_dir, 'model_document_topic.pt')
    model.to(topic_config.device)

    creterion = nn.CrossEntropyLoss()
    # lr_lambda = lambda epoch: 0.7 ** epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)  # 创建学习率调度器

    model.train()
    max_f = 0
    for epoch in range(topic_config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (sample, label) in enumerate(train_iter):
            sample = sample.to(topic_config.device)
            label = label.to(topic_config.device)
            logits = model(sample)
            loss = creterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            acc = (logits.argmax(1) == label).float().mean()
            # logging.info(f"Epoch: {epoch + 1}, Batch[{idx}/{len(train_iter)}], "
            #              f"Train loss :{loss.item():.4f}, Train acc: {acc:.4f}")
        # scheduler.step()  # 更新学习率
        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: {epoch + 1}, Train loss: {train_loss:.4f}, Epoch time = {(end_time - start_time):.4f}s")
        acc, p, r, f, _, _ = evaluate_document_topic(test_iter, model, topic_config.device)
        logging.info(f"Val: Epoch {epoch + 1} Accuracy: {acc:.4f},  Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")
        if f > max_f:
            max_f = f
            torch.save(model.state_dict(), model_save_path)


def evaluate_document_topic(data_iter, model, device):
    model.eval()
    with torch.no_grad():
        y_true, y_pred = [], []
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            batch_pred = logits.argmax(dim=1).tolist()  # 得到一个batch的预测标签
            batch_true = y.tolist()
            y_pred.extend(batch_pred)
            y_true.extend(batch_true)

        acc = accuracy_score(y_true, y_pred)
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f = f1_score(y_true, y_pred)

    model.train()
    return acc, p, r, f, y_true, y_pred


def inference_document_topic(topic_config):
    test_data = data_process_with_document_topics(topic_config.test_file_path, topic_config.test_document_topics_file_path)
    test_dataset = TopicDataset(test_data)
    test_iter = DataLoader(test_dataset, batch_size=topic_config.batch_size, shuffle=False,
                           collate_fn=generate_batch_with_document_topics)

    model = DocumentTopicModel(input_dim=topic_config.num_topics, num_labels=topic_config.num_labels)
    model_save_path = os.path.join(topic_config.topic_results_save_dir, 'model_document_topic.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info('## 成功载入已有模型，进行预测......')
    model.to(topic_config.device)
    acc, p, r, f, _, _ = evaluate_document_topic(test_iter, model, topic_config.device)
    logging.info(f"Test Results:    Accuracy: {acc:.4f},  Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")


# word topic
def train_word_topic(topic_config, word_topics):
    topic_embd = word_topics['topic_matrix']
    topic_matrix_tensor = torch.tensor(topic_embd, dtype=torch.float, device=topic_config.device)
    # 加载训练集、测试集
    train_data = data_process_with_word_topics(topic_config, topic_config.train_file_path,
                                               topic_config.segmented_train_file_path, word_topics)
    test_data = data_process_with_word_topics(topic_config, topic_config.test_file_path,
                                              topic_config.segmented_test_file_path, word_topics)
    train_dataset = TopicDataset(train_data)
    test_dataset = TopicDataset(test_data)
    train_iter = DataLoader(train_dataset, batch_size=topic_config.batch_size, shuffle=True,
                            collate_fn=generate_batch_with_word_topics)
    test_iter = DataLoader(test_dataset, batch_size=topic_config.batch_size, shuffle=False,
                           collate_fn=generate_batch_with_word_topics)

    model = WordTopicModel(input_dim=topic_config.num_topics, num_labels=topic_config.num_labels)
    model_save_path = os.path.join(topic_config.topic_results_save_dir, 'model_word_topic.pt')
    model.to(topic_config.device)

    creterion = nn.CrossEntropyLoss()

    # lr_lambda = lambda epoch: 0.7 ** epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)  # 创建学习率调度器

    model.train()
    max_f = 0
    for epoch in range(topic_config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (sample, label) in enumerate(train_iter):
            sample = sample.to(topic_config.device)
            label = label.to(topic_config.device)
            logits = model(sample, topic_matrix_tensor)
            loss = creterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            acc = (logits.argmax(1) == label).float().mean()
            # logging.info(f"Epoch: {epoch + 1}, Batch[{idx}/{len(train_iter)}], "
            #              f"Train loss :{loss.item():.4f}, Train acc: {acc:.4f}")
        # scheduler.step()  # 更新学习率
        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: {epoch + 1}, Train loss: {train_loss:.4f}, Epoch time = {(end_time - start_time):.4f}s")
        acc, p, r, f, _, _ = evaluate_word_topic(test_iter, model, topic_config.device, topic_matrix_tensor)
        logging.info(f"Val: Epoch {epoch + 1} Accuracy: {acc:.4f},  Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")
        if f > max_f:
            max_f = f
            torch.save(model.state_dict(), model_save_path)


def evaluate_word_topic(data_iter, model, device, topic_matrix):
    model.eval()
    with torch.no_grad():
        y_true, y_pred = [], []
        for sample, label in tqdm(data_iter, desc='Evaluating'):
            sample, label = sample.to(device), label.to(device)
            logits = model(sample, topic_matrix)
            batch_pred = logits.argmax(dim=1).tolist()  # 得到一个batch的预测标签
            batch_true = label.tolist()
            y_pred.extend(batch_pred)
            y_true.extend(batch_true)

        acc = accuracy_score(y_true, y_pred)
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f = f1_score(y_true, y_pred)

    model.train()
    return acc, p, r, f, y_true, y_pred


def inference_word_topic(topic_config, word_topics):
    topic_embd = word_topics['topic_matrix']
    topic_matrix_tensor = torch.tensor(topic_embd, dtype=torch.float, device=topic_config.device)

    test_data = data_process_with_word_topics(topic_config, topic_config.test_file_path,
                                              topic_config.segmented_test_file_path, word_topics)
    test_dataset = TopicDataset(test_data)
    test_iter = DataLoader(test_dataset, batch_size=topic_config.batch_size, shuffle=False,
                           collate_fn=generate_batch_with_word_topics)

    model = WordTopicModel(input_dim=topic_config.num_topics, num_labels=topic_config.num_labels)
    model_save_path = os.path.join(topic_config.topic_results_save_dir, 'model_word_topic.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info('## 成功载入已有模型，进行预测......')
    model.to(topic_config.device)
    acc, p, r, f, _, _ = evaluate_word_topic(test_iter, model, topic_config.device, topic_matrix_tensor)
    logging.info(f"Test Results:    Accuracy: {acc:.4f},  Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")


if __name__ == '__main__':
    torch.manual_seed(1234)
    topic_config = TopicConfig()
    # train_lda(topic_config)

    # document topics
    # train_document_topic(topic_config)
    # inference_document_topic(topic_config)

    # word topics
    # 加载lda训练好的主题-词向量
    word_topics = load_word_topics(topic_config, add_unk=True, recover_topic_peaks=False)
    train_word_topic(topic_config, word_topics)
    inference_word_topic(topic_config, word_topics)


