# 此模块用于进行数据的预处理操作，包括字符串替换、繁转简等数据清洗操作，分词，字典构建等等
import re
import os
import pandas as pd
from hanziconv import HanziConv
from collections import defaultdict  # 当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值
from tqdm.auto import tqdm
import jieba
from ltp import LTP
import torch

# from baseline.baseline_config import Config


class Preprocessor:
    """
    此类专门用于进行数据预处理
    """

    @staticmethod
    def basic_pipeline(s):
        """流程化处理"""
        s = Preprocessor.replace_invalid_char(s)
        s = Preprocessor.traditional_to_simplified(s)
        return s

    @staticmethod
    def replace_invalid_char(s):
        """
        将url链接、图片链接、@.. 替换为指定标记
        :param s: 一条样本 -> str
        :return:
        """
        url_token = '<URL>'
        img_token = '<IMG>'
        at_token = '<@ID>'
        s = re.sub(r'(http://)?www.*?(\s|$|[\u4e00-\u9fa5])', url_token + '\\2', s)  # URL containing www
        s = re.sub(r'http://.*?(\s|$|[\u4e00-\u9fa5])', url_token + '\\1', s)  # URL starting with http
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
    def build_vocab(config, tokenize=False):
        """
        词典实例化
        vocab.idx_to_token  # ['<unk>','希','望','大','家',...]
        vocab.token_to_idx  # {'<unk>': 0, '希': 1, '望': 2, '大': 3, '家': 4,...}
        vocab.convert_tokens_to_ids(['希','望'])  # [1, 2]
        vocab.convert_ids_to_tokens([1,2])  # ['希', '望']
        :param config: 超参数
        :param tokenize: 是否进行分词，默认为否
        :return: 实例化后的词典
        """
        print('正在构建词典...')
        train_data_filepath = config.train_data_filepath
        raw_iter = pd.read_csv(train_data_filepath)
        if tokenize:
            train_sentence = []
            for raw in raw_iter.values:
                s = raw[1]  # str
                s = Preprocessor.basic_pipeline(s)
                s = Preprocessor.process_for_segmented(s)
                train_sentence.append(s)
            sentence_tokenized = tokenize_ltp(train_sentence,
                                              user_dict_filepath=config.user_dict_filepath,
                                              filepath=config.segmented_train_filepath,
                                              postfix=config.tokenize_type)
            print('词典构建完成。')
            return Vocab.build(sentence_tokenized)

        train_sentence = [Preprocessor.basic_pipeline(raw[1]) for raw in raw_iter.values]
        print('词典构建完成。')
        return Vocab.build(train_sentence)

    @staticmethod
    def process_for_segmented(s):
        """数据处理,用于ltp分词"""
        s = re.sub(r'<@ID>|<URL>', '', s)
        s = re.sub(r'\s+', '，', s.strip())  # 将空白字符替换成逗号
        # s = re.sub(r'\W', '', s.strip())  # 过滤所有非字母数字下划线的字符
        return s

    @staticmethod
    def stop_words_list(stopwords_filepath):
        """加载停用词表"""
        stopwords = [line.strip() for line in open(stopwords_filepath, 'r', encoding='utf-8').readlines()]
        return stopwords

    @staticmethod
    def remove_stopwords(sentence, stopwords_filepath):
        """去停用词"""
        stopwords = Preprocessor.stop_words_list(stopwords_filepath)
        return [i for i in sentence if i not in stopwords]

    @staticmethod
    def remove_short_words(s):
        """过滤字符长度为1的词"""
        return [w for w in s if len(w) > 1]

    @staticmethod
    def remove_zero_words(s):
        """过滤空字符"""
        return [w for w in s if len(w) > 0]

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


class Vocab:
    def __init__(self, tokens=None):
        self.idx_to_token = list()  # 词表
        self.token_to_idx = dict()  # 词表及对应单词位置

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1  # 标记每个单词的位置
            self.unk = self.token_to_idx['<unk>']  # 开始符号的位置

    @classmethod
    # 不需要实例化，直接类名.方法名()来调用 不需要self参数，但第一个参数需要是表示自身类的cls参数,
    # 因为持有cls参数，可以来调用类的属性，类的方法，实例化对象等
    def build(cls, text, min_freq=1, reserved_tokens=None):
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token, freq in token_freqs.items() \
                        if freq >= min_freq and token != "<unk>"]
        return cls(uniq_tokens)

    def __len__(self):
        # 返回词表的大小，即词表中有多少个互不相同的标记
        return len(self.idx_to_token)

    def __getitem__(self, token):
        # 查找输入标记对应的索引值，如果该标记不存在，则返回标记<unk>的索引值（0）
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        # 查找一系列输入标记对应的索引值
        return [self[token] for token in tokens]

    def convert_ids_to_tokens(self, indices):
        # 查找一系列索引值对应的标记
        return [self.idx_to_token[index] for index in indices]


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
            print(f"缓存文件 {data_path} 不存在，重新处理并缓存！")
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            print(f"缓存文件 {data_path} 存在，直接载入缓存文件！")
            with open(data_path, 'rb') as f:
                data = torch.load(f)
        return data

    return wrapper


@cache
def tokenize_ltp(sentences, user_dict_filepath='./', filepath='./', postfix='cache'):
    """
    ltp中文分词
    :param sentences: 需要进行分词的句子列表  -> list
    :param user_dict_filepath: 自定义词典的位置
    :param filepath: 需要进行分词的数据集的位置，目的是为了保存缓存文件
    :param postfix: 缓存文件的标识
    :return: 分词结果 -> list
    """
    assert type(sentences) == list
    print('## 正在使用ltp分词工具进行分词')
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
    print('## 正在使用jieba分词工具进行分词')
    result = []
    Preprocessor.jieba_init(user_dict_filepath)
    for sentence in tqdm(sentences, desc='seg processing'):
        segment_jieba = jieba.cut(sentence)
        segment_jieba = [token for token in segment_jieba]
        # with open(filepath, 'a+', encoding='utf-8') as seg_file:
        #     seg_file.write(str(segment_jieba) + '\n')
        result.append(segment_jieba)
    return result


# if __name__ == '__main__':
#     config = Config()
#     vocab = Preprocessor.build_vocab(config, tokenize=True)
#     print(len(vocab))
#     print(vocab.token_to_idx)
