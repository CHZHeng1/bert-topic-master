import os
import logging
import time
import torch
from torch.optim import lr_scheduler
from transformers import BertTokenizer
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import ModelConfig
from utils.data_helpers import LoadSingleSentenceClassificationDataset
from model.down_stream_tasks.bert_classification import BertWithTopics


import sys
sys.path.append('../')

# 设置随机数种子
SEED = 1234
# random.seed(SEED)
# np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def train(config):
    # 1.训练lda -> 文档主题分布或主题词分布
    # 2.数据处理
    model = BertWithTopics(config, config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model_document_topic.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        logging.info("## 成功载入已有模型，进行追加训练......")
    model = model.to(config.device)

    lr_lambda = lambda epoch: 0.95 ** epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)  # 创建学习率调度器
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

    # BERT + document_topics
    train_iter, test_iter, val_iter = \
        data_loader.load_data_with_document_topics(train_file_path=config.train_file_path,
                                                   val_file_path=config.val_file_path,
                                                   test_file_path=config.test_file_path,
                                                   train_document_topics_file_path=config.train_document_topics_file_path,
                                                   val_document_topics_file_path=config.test_document_topics_file_path,
                                                   test_document_topics_file_path=config.test_document_topics_file_path
                                                   )
    max_f = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (sample, label, topic_vector) in enumerate(train_iter):
            sample = sample.to(config.device)  # [src_len, batch_size]
            label = label.to(config.device)
            topic_vector = topic_vector.to(config.device)
            padding_mask = (sample == data_loader.PAD_IDX).transpose(0, 1)
            loss, logits = model(
                input_ids=sample,
                attention_mask=padding_mask,
                token_type_ids=None,
                position_ids=None,
                labels=label,
                topic_vector=topic_vector)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            acc = (logits.argmax(1) == label).float().mean()
            # if idx % 10 == 0:
            logging.info(f"Epoch: {epoch+1}, Batch[{idx+1}/{len(train_iter)}], "
                         f"Train loss :{loss.item():.4f}, Train acc: {acc:.4f}")
        scheduler.step()
        end_time = time.time()
        # train_loss = losses / len(train_iter)
        train_loss = losses
        logging.info(f"Epoch: {epoch+1}, Train loss: {train_loss:.4f}, Epoch time = {(end_time - start_time):.4f}s")
        # if (epoch + 1) % config.model_val_per_epoch == 0:
        acc, p, r, f = evaluate(val_iter, model, config.device, data_loader.PAD_IDX)
        logging.info(f"Val: Epoch {epoch+1} Accuracy: {acc:.4f},  Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")
        if f > max_f:
            max_f = f
            torch.save(model.state_dict(), model_save_path)


def inference(config):
    model = BertWithTopics(config, config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, 'model_document_topic.pt')
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
        data_loader.load_data_with_document_topics(train_file_path=config.train_file_path,
                                                   val_file_path=config.val_file_path,
                                                   test_file_path=config.test_file_path,
                                                   train_document_topics_file_path=config.train_document_topics_file_path,
                                                   val_document_topics_file_path=config.test_document_topics_file_path,
                                                   test_document_topics_file_path=config.test_document_topics_file_path
                                                   )
    acc, p, r, f = evaluate(test_iter, model, config.device, data_loader.PAD_IDX)
    logging.info(f"Test Results:    Accuracy: {acc:.4f},  Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")


def evaluate(data_iter, model, device, PAD_IDX):
    model.eval()
    with torch.no_grad():
        test_losses = 0
        y_true, y_pred = [], []
        for x, y, z in tqdm(data_iter, desc='Evaluating:'):
            x, y, z = x.to(device), y.to(device), z.to(device)
            padding_mask = (x == PAD_IDX).transpose(0, 1)
            loss, logits = model(x, attention_mask=padding_mask,
                                 labels=y,
                                 topic_vector=z)
            test_losses += loss.item()
            batch_pred = logits.argmax(dim=1).tolist()  # 得到一个batch的预测标签
            batch_true = y.tolist()
            y_pred.extend(batch_pred)
            y_true.extend(batch_true)

        logging.info(f'Test Loss:{test_losses:.4f}')
        acc = accuracy_score(y_true, y_pred)
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f = f1_score(y_true, y_pred)

        model.train()
        return acc, p, r, f


if __name__ == '__main__':
    model_config = ModelConfig()
    train(model_config)
    inference(model_config)

