import os
import logging
import time
import csv
import torch
from torch.optim import lr_scheduler
from transformers import BertTokenizer
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import ModelConfig
from utils.data_helpers import LoadSingleSentenceClassificationDataset
from topic_model.topic_loader import load_word_topics
from model.down_stream_tasks.bert_classification import BertTopic

# 设置随机数种子
SEED = 1234
# random.seed(SEED)
# np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def train(config, feature_name=None):
    model = BertTopic(config, feature_name=feature_name, bert_pretrained_model_dir=config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, f'{feature_name}.pt')
    # if os.path.exists(model_save_path):
    #     loaded_paras = torch.load(model_save_path)
    #     model.load_state_dict(loaded_paras)
    #     logging.info("## 成功载入已有模型，进行追加训练......")
    model = model.to(config.device)
    # lr_lambda = lambda epoch: 0.9 ** epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
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

    if feature_name == 'BERT-W-I':
        train_iter, val_iter = \
            data_loader.load_data_for_ablation_study(config, only_test=False, feature_name=feature_name,
                                                     collate_fn=data_loader.generate_batch_for_bert_w_i)
    elif feature_name == 'BERT-W-R':
        train_iter, val_iter = \
            data_loader.load_data_for_ablation_study(config, only_test=False, feature_name=feature_name,
                                                     collate_fn=data_loader.generate_batch_for_bert_w_r)
    elif feature_name == 'BERT-I-R':
        train_iter, val_iter = \
            data_loader.load_data_for_ablation_study(config, only_test=False, feature_name=feature_name,
                                                     collate_fn=data_loader.generate_batch_for_bert_i_r)
    elif feature_name == 'W-I-R':
        train_iter, val_iter = \
            data_loader.load_data_for_ablation_study(config, only_test=False, feature_name=feature_name,
                                                     collate_fn=data_loader.generate_batch_for_w_i_r)
    else:
        raise TypeError

    word_topics = load_word_topics(config, add_unk=True, recover_topic_peaks=False)
    word_topics_embedding = torch.tensor(word_topics['topic_matrix'], dtype=torch.float, device=config.device)

    max_f = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, batch in enumerate(train_iter):
            if feature_name == 'BERT-W-I':
                bert_sample, word_sample, feature_influence, labels = batch
                bert_sample = bert_sample.to(config.device)
                word_sample = word_sample.to(config.device)
                feature_influence = feature_influence.to(config.device)
                labels = labels.to(config.device)
                padding_mask = (bert_sample == data_loader.PAD_IDX).transpose(0, 1)

                loss, logits = model(input_ids=bert_sample, attention_mask=padding_mask, labels=labels,
                                     word_topics_id=word_sample, word_topics_embedding=word_topics_embedding,
                                     feature_influence=feature_influence)

            elif feature_name == 'BERT-W-R':
                bert_sample, word_sample, feature_reliability, labels = batch
                bert_sample = bert_sample.to(config.device)
                word_sample = word_sample.to(config.device)
                feature_reliability = feature_reliability.to(config.device)
                labels = labels.to(config.device)
                padding_mask = (bert_sample == data_loader.PAD_IDX).transpose(0, 1)

                loss, logits = model(input_ids=bert_sample, attention_mask=padding_mask, labels=labels,
                                     word_topics_id=word_sample, word_topics_embedding=word_topics_embedding,
                                     feature_reliability=feature_reliability)

            elif feature_name == 'BERT-I-R':
                bert_sample, feature_influence, feature_reliability, labels = batch
                bert_sample = bert_sample.to(config.device)
                feature_influence = feature_influence.to(config.device)
                feature_reliability = feature_reliability.to(config.device)
                labels = labels.to(config.device)
                padding_mask = (bert_sample == data_loader.PAD_IDX).transpose(0, 1)

                loss, logits = model(input_ids=bert_sample, attention_mask=padding_mask, labels=labels,
                                     feature_influence=feature_influence, feature_reliability=feature_reliability)

            elif feature_name == 'W-I-R':
                word_sample, feature_influence, feature_reliability, labels = batch
                word_sample = word_sample.to(config.device)
                feature_influence = feature_influence.to(config.device)
                feature_reliability = feature_influence.to(config.device)
                labels = labels.to(config.device)
                loss, logits = model(labels=labels, word_topics_id=word_sample,
                                     word_topics_embedding=word_topics_embedding,
                                     feature_influence=feature_influence,
                                     feature_reliability=feature_reliability)
            else:
                raise TypeError

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            acc = (logits.argmax(1) == labels).float().mean()
            # if idx % 10 == 0:
            logging.info(f"Epoch: {epoch+1}, Batch[{idx+1}/{len(train_iter)}], "
                         f"Train loss :{loss.item():.4f}, Train acc: {acc:.4f}")
        # scheduler.step()
        end_time = time.time()
        # train_loss = losses / len(train_iter)
        train_loss = losses
        logging.info(f"Epoch: {epoch+1}, Train loss: {train_loss:.4f}, Epoch time = {(end_time - start_time):.4f}s")
        # if (epoch + 1) % config.model_val_per_epoch == 0:
        acc, p, r, f, _, _ = evaluate(val_iter, model, config.device,
                                      data_loader.PAD_IDX, feature_name, word_topics_embedding)
        logging.info(f"Val: Epoch {epoch+1} Accuracy: {acc:.4f},  Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")
        if f > max_f:
            max_f = f
            torch.save(model.state_dict(), model_save_path)


def evaluate(data_iter, model, device, PAD_IDX, feature_name=None, word_topics_embedding=None):
    model.eval()
    with torch.no_grad():
        y_true, y_pred = [], []
        test_losses = 0
        for batch in tqdm(data_iter, desc='Evaluating'):
            if feature_name == 'BERT-W-I':
                bert_sample, word_sample, feature_influence, labels = batch
                bert_sample = bert_sample.to(device)
                word_sample = word_sample.to(device)
                feature_influence = feature_influence.to(device)
                labels = labels.to(device)
                padding_mask = (bert_sample == PAD_IDX).transpose(0, 1)

                loss, logits = model(input_ids=bert_sample, attention_mask=padding_mask, labels=labels,
                                     word_topics_id=word_sample, word_topics_embedding=word_topics_embedding,
                                     feature_influence=feature_influence)

            elif feature_name == 'BERT-W-R':
                bert_sample, word_sample, feature_reliability, labels = batch
                bert_sample = bert_sample.to(device)
                word_sample = word_sample.to(device)
                feature_reliability = feature_reliability.to(device)
                labels = labels.to(device)
                padding_mask = (bert_sample == PAD_IDX).transpose(0, 1)

                loss, logits = model(input_ids=bert_sample, attention_mask=padding_mask, labels=labels,
                                     word_topics_id=word_sample, word_topics_embedding=word_topics_embedding,
                                     feature_reliability=feature_reliability)

            elif feature_name == 'BERT-I-R':
                bert_sample, feature_influence, feature_reliability, labels = batch
                bert_sample = bert_sample.to(device)
                feature_influence = feature_influence.to(device)
                feature_reliability = feature_reliability.to(device)
                labels = labels.to(device)
                padding_mask = (bert_sample == PAD_IDX).transpose(0, 1)

                loss, logits = model(input_ids=bert_sample, attention_mask=padding_mask, labels=labels,
                                     feature_influence=feature_influence, feature_reliability=feature_reliability)

            elif feature_name == 'W-I-R':
                word_sample, feature_influence, feature_reliability, labels = batch
                word_sample = word_sample.to(device)
                feature_influence = feature_influence.to(device)
                feature_reliability = feature_influence.to(device)
                labels = labels.to(device)
                loss, logits = model(labels=labels, word_topics_id=word_sample,
                                     word_topics_embedding=word_topics_embedding,
                                     feature_influence=feature_influence,
                                     feature_reliability=feature_reliability)
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


def inference(config, feature_name=None):
    model = BertTopic(config, feature_name=feature_name, bert_pretrained_model_dir=config.pretrained_model_dir)
    model_save_path = os.path.join(config.model_save_dir, f'{feature_name}.pt')
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

    if feature_name == 'BERT-W-I':
        test_iter = \
            data_loader.load_data_for_ablation_study(config, only_test=True, feature_name=feature_name,
                                                     collate_fn=data_loader.generate_batch_for_bert_w_i)
    elif feature_name == 'BERT-W-R':
        test_iter = \
            data_loader.load_data_for_ablation_study(config, only_test=True, feature_name=feature_name,
                                                     collate_fn=data_loader.generate_batch_for_bert_w_r)
    elif feature_name == 'BERT-I-R':
        test_iter = \
            data_loader.load_data_for_ablation_study(config, only_test=True, feature_name=feature_name,
                                                     collate_fn=data_loader.generate_batch_for_bert_i_r)
    elif feature_name == 'W-I-R':
        test_iter = \
            data_loader.load_data_for_ablation_study(config, only_test=True, feature_name=feature_name,
                                                     collate_fn=data_loader.generate_batch_for_w_i_r)
    else:
        raise TypeError

    word_topics = load_word_topics(config, add_unk=True, recover_topic_peaks=False)
    word_topics_embedding = torch.tensor(word_topics['topic_matrix'], dtype=torch.float, device=config.device)

    acc, p, r, f, y_true, y_pred = evaluate(test_iter, model, config.device,
                                            data_loader.PAD_IDX, feature_name, word_topics_embedding)
    logging.info(f"Test Results:    Accuracy: {acc:.4f},  Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")
    # for i in range(len(y_true)):
    #     label = [y_true[i], y_pred[i]]
    #     save_csv('labels_bert', label)


def save_csv(filename, content):
    fp = open(f"{filename}.csv", 'a+', encoding='utf-8-sig', newline='')
    csv_fp = csv.writer(fp)
    csv_fp.writerow(content)
    fp.close()
    # print(f"成功写入：{content}")


if __name__ == '__main__':
    model_config = ModelConfig()
    train(model_config, feature_name='BERT-I-R')
    inference(model_config, feature_name='BERT-I-R')
