import os
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from gensim.models import Word2Vec
import torch
from torch import nn
from torch.nn import functional as F

from baseline.baseline_config import Config
from baseline.utils.data_preprocessor import Preprocessor, tokenize_ltp


# 此模块共包含以下内容：
# 数据处理
# 训练模型
# 保存模型
# 加载模型
# 保存训练好的词向量
# 加载词向量


def train_save_word2vec_model(config, sentences, save_model=False):
    """训练word2vec模型
       目前设置的参数是基于负采样的Skip-gram模型
    :param config: 超参数
    :param sentences: 分词完成后的样本 sentences[0] -> list  sentences[0][0] -> str (token)
    :param save_model: 是否保存训练好的模型
    """
    word2vec_model = Word2Vec(sentences=sentences,  #
                              vector_size=config.vector_dim,  # 生成词向量维度
                              window=5,  # 上下文窗口大小
                              sg=1,  # sg=0时，为CBOW模型，sg=1时，为skip-gram模型
                              hs=0,  # 0时为负采样，1时且negative大于0，则为Hierarchical Softmax
                              negative=5,  # 负采样时，负采样的个数，建议在[3,10]之间
                              min_count=2,  # 需要计算词向量的最小词频
                              epochs=10,  # 迭代次数
                              # alpha=0.025,  # 初始学习率
                              # min_alpha=5e-5,  # 学习率最小值
                              seed=1234
                              )
    if save_model:
        model_path = os.path.join(config.word2vec_save_dir, 'word2vec_model')
        print(f'Saving Word2vec Model to {model_path}...')
        word2vec_model.save(model_path)
    return word2vec_model


def load_word2vec_model(config):
    """加载训练好的word2vec模型"""
    model_path = os.path.join(config.word2vec_save_dir, 'word2vec_model')
    print(f'Loading Word2vec Model from {model_path}...')
    word2vec_model = Word2Vec.load(model_path)
    return word2vec_model


def infer_and_write_word2vec_vector(config, word2vec_model=None, max_vocab=None):
    """
    推断并保存词向量
    :param config: 超参数
    :param word2vec_model: 训练好的模型
    :param max_vocab: 词典最多容纳的单词数量
    """
    if word2vec_model is None:
        word2vec_model = load_word2vec_model(config)
    print('Inferring and writing word2vec vector...')
    if max_vocab is None:
        vocab_len = len(word2vec_model.wv.index_to_key)
    else:
        vocab_len = max_vocab
    print(f'vocab_size: {vocab_len}')
    word2vec_vector_file = os.path.join(config.word2vec_save_dir, 'word2vec_vector.log')
    with open(word2vec_vector_file, 'w', encoding='utf-8') as outfile:
        model_path = os.path.join(config.word2vec_save_dir, 'word2vec_model')
        outfile.writelines(f'Loading Word2vec Model from {model_path}\n')
        outfile.writelines(f'{vocab_len}\n')
        for i in tqdm(range(vocab_len), desc='writing'):
            w = word2vec_model.wv.index_to_key[i]  # word
            if max_vocab is None or i < max_vocab:
                # \s: 匹配任何空白字符,等价于[ \f\n\r\t\v] +: 匹配前面的子表达式一次或多次
                vector_str = re.sub('\s+', ' ', str(word2vec_model.wv[i]))
                # 去除 ] 前的空白字符
                vector_str = re.sub('\s+]', ']', str(vector_str))
                vector_str = vector_str.replace('[ ', '[')
                line = '{} {} {}'.format(i, w, vector_str)
                outfile.writelines(line + '\n')
            else:
                break
    print(f'Word2vec vector saved to {word2vec_vector_file}')


def load_word2vec_vector(config, add_unk=True):
    """
    加载训练好的词向量
    :param config: 超参数
    :param add_unk: 是否添加未登录词的处理
    """
    complete_word_vector_dict = {}
    id_word_dict = {}
    word_id_dict = {}
    count = 0
    vector_matrix = []
    vector_dim = config.vector_dim
    word2vec_vector_file = os.path.join(config.word2vec_save_dir, 'word2vec_vector.log')
    print(f"Reading word2vec vector from {word2vec_vector_file}")

    if add_unk:  # 处理未登录词
        word = '<unk>'
        word_vector = np.array([0.0]*vector_dim)

        word_id = 0
        id_word_dict[word_id] = word
        word_id_dict[word] = word_id
        complete_word_vector_dict[word] = word_vector
        vector_matrix.append(word_vector)

    # 读取训练好的词向量
    with open(word2vec_vector_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            count += 1
            if count > 2:
                ldata = line.rstrip().split(' ')  # 以空格分割
                if add_unk:
                    word_id = int(ldata[0]) + 1  # 序号 因为前面有一个未登录词，所以这里要加1
                else:
                    word_id = int(ldata[0])
                word = ldata[1]  # 词
                id_word_dict[word_id] = word  # {id:word,...}
                word_id_dict[word] = word_id  # {word:id,...}
                word_vector = np.array([float(s.replace('[', '').replace(']', '')) for s in ldata[2:]])
                assert len(word_vector) == vector_dim
                complete_word_vector_dict[word] = word_vector
                vector_matrix.append(word_vector)

    vector_matrix = np.array(vector_matrix)
    print(f'word vector embedding dim: {vector_matrix.shape}')
    assert len(vector_matrix.shape) == 2
    return {'complete_word_dict': complete_word_vector_dict, 'id2word_dict': id_word_dict,
            'word2id_dict': word_id_dict, 'vector_matrix': vector_matrix}


def word2vec_process(config):
    """数据处理"""
    train_file_path = config.train_data_filepath
    raw_iter = pd.read_csv(train_file_path)
    sentences = []
    for raw in tqdm(raw_iter.values, desc='Data Processing'):
        s = raw[1]  # 文本
        s = Preprocessor.basic_pipeline(s)
        s = Preprocessor.process_for_segmented(s)
        sentences.append(s)

    data_tokenized = tokenize_ltp(sentences, user_dict_filepath=config.user_dict_filepath,
                                  filepath=config.segmented_train_filepath, postfix=config.tokenize_type)
    # 过滤非字母数据下划线的字符
    data_finished = [Preprocessor.remove_special_words(s) for s in data_tokenized]
    # 过滤空字符
    data_finished = [Preprocessor.remove_zero_words(s) for s in data_finished]
    # word2vec基于上下文之间的共现关系，因此不去停用词
    return data_finished


class Word2vecLDA(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=512, num_class=2):
        super(Word2vecLDA, self).__init__()
        self.full_connection_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_class)
        self.normalize = nn.LayerNorm(hidden_dim)
        self.activate = torch.relu

    def forward(self, word_vector, topic_vector):
        inputs = torch.cat([word_vector, topic_vector], dim=1)
        fc_layer_out = self.normalize(self.full_connection_layer(inputs))
        fc_layer_out = self.activate(fc_layer_out)
        hidden_layer_out = self.normalize(self.hidden_layer(fc_layer_out))
        pooled_out = self.activate(hidden_layer_out)
        # pooled_out = F.dropout(hidden_layer_out, 0.5)
        outputs = self.classifier(pooled_out)
        return outputs


# if __name__ == '__main__':
#     # wv_config = Config()
#     # train_data_tokenized = word2vec_process(wv_config)
#     # wv_model = train_save_word2vec_model(wv_config, sentences=train_data_tokenized, save_model=True)
#     # infer_and_write_word2vec_vector(wv_config, wv_model, max_vocab=None)
#
#     word_vector_table = load_word2vec_vector(wv_config, add_unk=True)
#     unk_vector = word_vector_table['vector_matrix'][0]
#     print(unk_vector)





