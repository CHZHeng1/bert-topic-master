import os
import logging
import time
import csv
import torch
from torch.optim import lr_scheduler
from transformers import BertTokenizer
from tqdm.auto import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import ModelConfig
from utils.data_helpers import LoadSingleSentenceClassificationDataset
from topic_model.topic_loader import load_word_topics
from model.down_stream_tasks.bert_classification import BertWithWordWeiboUser


import sys
sys.path.append('../')

# 设置随机数种子
SEED = 1234
# random.seed(SEED)
# np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def save_csv(filename, content):
    fp = open(f"{filename}.csv", 'a+', encoding='utf-8-sig', newline='')
    csv_fp = csv.writer(fp)
    csv_fp.writerow(content)
    fp.close()
    # print(f"成功写入：{content}")


def train(config):
    # 加载主题词表
    word_topics = load_word_topics(config, add_unk=True, recover_topic_peaks=False)
    topic_embd = word_topics['topic_matrix']
    topic_matrix_tensor = torch.tensor(topic_embd, dtype=torch.float, device=config.device)

    model = BertWithWordWeiboUser(config, config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'bert_word_weibo_user.pt')
    # if os.path.exists(model_save_path):
    #     loaded_paras = torch.load(model_save_path)
    #     model.load_state_dict(loaded_paras)
    #     logging.info("## 成功载入已有模型，进行追加训练......")
    model = model.to(config.device)

    # lr_lambda = lambda epoch: 0.95 ** epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)  # 创建学习率调度器

    model.train()
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
    data_loader = LoadSingleSentenceClassificationDataset(vocab_path=config.vocab_path,
                                                          tokenizer=bert_tokenize,
                                                          batch_size=config.batch_size,
                                                          max_sen_len=config.max_sen_len,
                                                          split_sep=config.split_sep,
                                                          max_position_embeddings=config.max_position_embeddings,
                                                          pad_index=config.pad_token_id,
                                                          is_sample_shuffle=config.is_sample_shuffle)

    # BERT + word_topics
    train_iter, test_iter, val_iter = \
        data_loader.load_data_with_word_and_ir(config,
                                               word_topics=word_topics,
                                               train_file_path=config.train_file_path,
                                               val_file_path=config.val_file_path,
                                               test_file_path=config.test_file_path,
                                               train_manual_feature_file_path=config.train_manual_features_file_path,
                                               val_manual_feature_file_path=config.test_manual_features_file_path,
                                               test_manual_feature_file_path=config.test_manual_features_file_path,
                                               only_test=False)

    max_f = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (sample_bert, label, sample_topic, feature_variable) in enumerate(train_iter):
            sample_bert = sample_bert.to(config.device)  # [src_len, batch_size]
            label = label.to(config.device)
            sample_topic = sample_topic.to(config.device)  # [batch_size, topic_nums]
            feature_variable = feature_variable.to(config.device)
            padding_mask = (sample_bert == data_loader.PAD_IDX).transpose(0, 1)
            loss, logits = model(input_ids=sample_bert,
                                 attention_mask=padding_mask,
                                 token_type_ids=None,
                                 position_ids=None,
                                 labels=label,
                                 word_topic_ids=sample_topic,  # [batch_size, max_src_len]
                                 feature_variable=feature_variable,
                                 topic_matrix=topic_matrix_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            acc = (logits.argmax(1) == label).float().mean()
            # if idx % 10 == 0:
            logging.info(f"Epoch: {epoch+1}, Batch[{idx+1}/{len(train_iter)}], "
                         f"Train loss :{loss.item():.4f}, Train acc: {acc:.4f}")

        # scheduler.step()  # 更新学习率
        end_time = time.time()
        # train_loss = losses / len(train_iter)
        train_loss = losses
        logging.info(f"Epoch: {epoch+1}, Train loss: {train_loss:.4f}, Epoch time = {(end_time - start_time):.4f}s")
        # if (epoch + 1) % config.model_val_per_epoch == 0:
        acc, p, r, f, _, _ = evaluate(val_iter, model, config.device, data_loader.PAD_IDX, topic_matrix_tensor)
        logging.info(f"Val: Epoch {epoch+1} Accuracy: {acc:.4f},  Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")
        if f > max_f:
            max_f = f
            torch.save(model.state_dict(), model_save_path)


def inference(config):
    # 加载主题词表
    word_topics = load_word_topics(config, add_unk=True, recover_topic_peaks=False)
    topic_embd = word_topics['topic_matrix']
    topic_matrix_tensor = torch.tensor(topic_embd, dtype=torch.float, device=config.device)

    model = BertWithWordWeiboUser(config, config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'bert_word_weibo_user.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行预测......")
    model = model.to(config.device)
    data_loader = LoadSingleSentenceClassificationDataset(vocab_path=config.vocab_path,
                                                          tokenizer=BertTokenizer.from_pretrained(
                                                              config.pretrained_model_dir).tokenize,
                                                          batch_size=config.batch_size,
                                                          max_sen_len=config.max_sen_len,
                                                          split_sep=config.split_sep,
                                                          max_position_embeddings=config.max_position_embeddings,
                                                          pad_index=config.pad_token_id,
                                                          is_sample_shuffle=config.is_sample_shuffle)
    train_iter, test_iter, val_iter = \
        data_loader.load_data_with_word_and_ir(config,
                                               word_topics,
                                               train_file_path=config.train_file_path,
                                               val_file_path=config.val_file_path,
                                               test_file_path=config.test_file_path,
                                               train_manual_feature_file_path=config.train_manual_features_file_path,
                                               val_manual_feature_file_path=config.test_manual_features_file_path,
                                               test_manual_feature_file_path=config.test_manual_features_file_path,
                                               only_test=False)

    acc, p, r, f, y_true, y_pred = evaluate(test_iter, model, config.device, data_loader.PAD_IDX, topic_matrix_tensor)
    logging.info(f"Test Results:    Accuracy: {acc:.4f},  Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")
    # for i in range(len(y_true)):
    #     label = [y_true[i], y_pred[i]]
    #     save_csv('labels_word', label)


def evaluate(data_iter, model, device, PAD_IDX, topic_matrix_tensor):
    model.eval()
    with torch.no_grad():
        y_true, y_pred = [], []
        test_losses = 0
        for sample_bert, label, sample_topic, feature_variable in tqdm(data_iter, desc='Evaluating'):
            sample_bert = sample_bert.to(device)
            label = label.to(device)
            sample_topic = sample_topic.to(device)
            feature_variable = feature_variable.to(device)
            padding_mask = (sample_bert == PAD_IDX).transpose(0, 1)

            loss, logits = model(input_ids=sample_bert,
                                 attention_mask=padding_mask,
                                 labels=label,
                                 word_topic_ids=sample_topic,
                                 feature_variable=feature_variable,
                                 topic_matrix=topic_matrix_tensor)

            test_losses += loss.item()

            batch_pred = logits.argmax(dim=1).tolist()  # 得到一个batch的预测标签
            batch_true = label.tolist()
            y_pred.extend(batch_pred)
            y_true.extend(batch_true)

        logging.info(f'Test Loss:{test_losses:.4f}')
        acc = accuracy_score(y_true, y_pred)
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f = f1_score(y_true, y_pred)

        model.train()
        return acc, p, r, f, y_true, y_pred


if __name__ == '__main__':
    model_config = ModelConfig()
    # train(model_config)
    inference(model_config)

