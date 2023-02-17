import os
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.optim import lr_scheduler

from utils.data_process import FakeNewsDataset, CollateFn
from utils.metrics import cal_precision, cal_recall, cal_f1
from model.text_cnn import TextCNN
from model.lstm import LSTM
from model.word2vec import Word2vecLDA
from model.machine_learning_methods import LogisticRegression
from baseline_config import Config


def train(config, model_name=None):
    # 数据
    data_loader = FakeNewsDataset(config, model_name=model_name)
    if model_name == 'TextCNN':
        model_save_path = os.path.join(config.model_save_dir, 'mode_text_cnn.pt')  # 模型保存位置
        train_iter, val_iter = data_loader.load_data(only_test=False,
                                                     collate_fn=CollateFn.generate_batch_textcnn,
                                                     model_name=model_name)
        # 模型
        model = TextCNN(vocab_size=data_loader.vocab_size, embedding_dim=config.embedding_dim,
                        filter_size=config.filter_size, num_filter=config.num_filter, num_class=config.num_class)

    elif model_name == 'Bi-LSTM':
        model_save_path = os.path.join(config.model_save_dir, 'mode_lstm.pt')  # 模型保存位置
        train_iter, val_iter, = data_loader.load_data(only_test=False,
                                                      collate_fn=CollateFn.generate_batch_lstm,
                                                      model_name=model_name)
        model = LSTM(vocab_size=data_loader.vocab_size, embedding_dim=config.embedding_dim,
                     hidden_dim=config.hidden_dim, num_class=config.num_class)

    elif model_name == 'Word2vecLDA':
        model_save_path = os.path.join(config.model_save_dir, 'mode_word2vec+lda.pt')  # 模型保存位置
        train_iter, val_iter = data_loader.load_data(only_test=False,
                                                     collate_fn=CollateFn.generate_batch_word2vec,
                                                     model_name=model_name)
        model = Word2vecLDA(input_dim=config.vector_dim + config.topic_dim,
                            hidden_dim=config.hidden_dim,
                            num_class=config.num_class)

    elif model_name == 'LogisticRegression':
        model_save_path = os.path.join(config.model_save_dir, 'mode_logistic_regression.pt')
        train_iter, val_iter = data_loader.load_data(only_test=False,
                                                     collate_fn=CollateFn.generate_batch_logistic_regression,
                                                     model_name=model_name)
        model = LogisticRegression(input_dim=config.topic_dim+config.manual_features_dim,
                                   num_class=config.num_class)

    else:
        print('模型名称输入有误。')
        return

    model = model.to(config.device)
    creterion = nn.CrossEntropyLoss()
    lr_lambda = lambda epoch: 0.95 ** epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)  # 创建学习率调度器

    model.train()

    # 训练
    max_f = 0
    for epoch in range(config.epochs):
        total_loss = 0
        total_acc = 0
        labels = []  # 整个数据集上的标签
        for batch in tqdm(train_iter, desc=f'Training Epoch {epoch + 1}'):
            probs, targets = None, None
            if model_name == 'TextCNN':
                inputs, targets = [x.to(config.device) for x in batch]  # 将数据加载至GPU
                probs = model(inputs)  # 将特征带入到模型

            elif model_name == 'Bi-LSTM':
                inputs, lengths, targets = [x.to(config.device) for x in batch]
                probs = model(inputs, lengths)

            elif model_name == 'Word2vecLDA':
                word_vector, topic_vector, targets = [x.to(config.device) for x in batch]
                probs = model(word_vector, topic_vector)

            elif model_name == 'LogisticRegression':
                inputs, targets = [x.to(config.device) for x in batch]
                probs = model(inputs)

            # 计算损失
            loss = creterion(probs, targets)
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            acc = (probs.argmax(dim=1) == targets).sum().item()  # item()用于在只包含一个元素的tensor中提取值
            total_acc += acc  # 最终得到整个epoch的准确率
            total_loss += loss.item()  # 最终得到整个epoch的损失

            batch_labels = targets.tolist()
            labels.extend(batch_labels)

        scheduler.step()  # 更新学习率
        # 打印的是整个eopch上的样本损失的平均值以及准确率
        print(f'Train Loss:{total_loss:.4f}    Train Accuracy:{total_acc/len(labels):.4f}')

        val_loss, acc, p, r, f, _, _ = evaluate(model, creterion, val_iter, config.device, model_name)
        print(f'Val Loss:{val_loss:.4f}    Val Accuracy:{acc:.4f}')
        print(f"Val Results:    Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")
        if f > max_f:
            max_f = f
            torch.save(model.state_dict(), model_save_path)


def evaluate(model, creterion, data_iter, device, model_name=None):
    test_acc = 0
    test_loss = 0
    model.eval()  # 切换到测试模式
    with torch.no_grad():  # 不计算梯度
        y_true, y_pred = [], []
        for batch in data_iter:
            if model_name == 'TextCNN':
                inputs, targets = [x.to(device) for x in batch]
                probs = model(inputs)

            elif model_name == 'Bi-LSTM':
                inputs, lengths, targets = [x.to(device) for x in batch]
                probs = model(inputs, lengths)

            elif model_name == 'Word2vecLDA':
                word_vector, topic_vector, targets = [x.to(device) for x in batch]
                probs = model(word_vector, topic_vector)

            elif model_name == 'LogisticRegression':
                inputs, targets = [x.to(device) for x in batch]
                probs = model(inputs)

            loss = creterion(probs, targets)
            acc = (probs.argmax(dim=1) == targets).sum().item()
            test_acc += acc
            test_loss += loss.item()

            batch_pred = probs.argmax(dim=1).tolist()  # 得到一个batch的预测标签
            batch_true = targets.tolist()

            y_pred.extend(batch_pred)
            y_true.extend(batch_true)

        acc = test_acc / len(y_true)
        p = cal_precision(y_true, y_pred)
        r = cal_recall(y_true, y_pred)
        f = cal_f1(y_true, y_pred)
    model.train()  # 切换到训练模式
    return test_loss, acc, p, r, f, y_true, y_pred


def predict(config, model_name=None):
    """模型效果预测"""
    data_loader = FakeNewsDataset(config, model_name=model_name)

    if model_name == 'TextCNN':
        test_iter = data_loader.load_data(only_test=True, collate_fn=CollateFn.generate_batch_textcnn,
                                          model_name=model_name)
        model = TextCNN(vocab_size=data_loader.vocab_size, embedding_dim=config.embedding_dim,
                        filter_size=config.filter_size, num_filter=config.num_filter, num_class=config.num_class)
        model_save_path = os.path.join(config.model_save_dir, 'mode_text_cnn.pt')

    elif model_name == 'Bi-LSTM':
        test_iter = data_loader.load_data(only_test=True, collate_fn=CollateFn.generate_batch_lstm,
                                          model_name=model_name)
        model = LSTM(vocab_size=data_loader.vocab_size, embedding_dim=config.embedding_dim,
                     hidden_dim=config.hidden_dim, num_class=config.num_class)
        model_save_path = os.path.join(config.model_save_dir, 'mode_lstm.pt')

    elif model_name == 'Word2vecLDA':
        test_iter = data_loader.load_data(only_test=True, collate_fn=CollateFn.generate_batch_word2vec,
                                          model_name=model_name)
        model = Word2vecLDA(input_dim=config.vector_dim + config.topic_dim,
                            hidden_dim=config.hidden_dim,
                            num_class=config.num_class)
        model_save_path = os.path.join(config.model_save_dir, 'mode_word2vec+lda.pt')

    elif model_name == 'LogisticRegression':
        test_iter = data_loader.load_data(only_test=True, collate_fn=CollateFn.generate_batch_logistic_regression,
                                          model_name=model_name)
        model_save_path = os.path.join(config.model_save_dir, 'mode_logistic_regression.pt')
        model = LogisticRegression(input_dim=config.topic_dim + config.manual_features_dim,
                                   num_class=config.num_class)
    else:
        print('模型名称输入有误。')
        return

    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        model.load_state_dict(loaded_paras)
        print('成功载入已有模型，进行预测......')
    model = model.to(config.device)
    # print(list(model.modules()))
    creterion = nn.CrossEntropyLoss()
    test_loss, acc, p, r, f, _, _ = evaluate(model, creterion, test_iter, config.device, model_name)

    print(f'Test Loss:{test_loss:.4f}    Test Accuracy:{acc:.4f}')
    print(f"Test Results:    Precision: {p:.4f},  Recall: {r:.4f},  F1: {f:.4f}")


if __name__ == '__main__':
    # 已知bug： word2vec + lda 和 Logistic Regression gpu运行报错
    torch.manual_seed(1234)
    model_config = Config()
    train(model_config, model_name='LogisticRegression')
    predict(model_config, model_name='LogisticRegression')
