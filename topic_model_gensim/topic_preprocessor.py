import re
import numpy as np
import nltk
from ltp import LTP
from tqdm.auto import tqdm
from hanziconv import HanziConv

from utils.data_helpers import cache


class Preprocessor:
    '''数据处理模块'''
    vocab_processor = None

    @staticmethod
    def basic_pipeline(sentences):
        '''
        基本数据清洗，bert和ltp分词的必要步骤
        sentence -> list
        sentence[0] -> str
        '''
        # process text
        print("Preprocessor: replace urls,@id,invalid...")
        sentences = Preprocessor.replace_url_id_invalid(sentences)
        print("Preprocessor: traditional to simplified")
        sentences = Preprocessor.traditional_to_simplified(sentences)

        # print("Preprocessor: split sentence into words")
        # sentences = Preprocessor.tokenize_tweet(sentences)
        # print("Preprocessor: remove quotes")
        # sentences = Preprocessor.removeQuotes(sentences)
        return sentences

    @staticmethod
    def replace_url_id_invalid(sentences):
        out = []
        # URL_tokens = ['<url>','<URL>','URLTOK']  # 'URLTOK' or '<URL>'
        # IMG_tokens = ['<pic>','IMG']
        URL_token = '<URL>'
        IMG_token = '<IMG>'
        AT_token = '<@ID>'
        for s in sentences:
            s = re.sub(r'(http://)?www.*?(\s|$)', URL_token+'\\2', s) # URL containing www
            s = re.sub(r'http://.*?(\s|$)', URL_token+'\\1', s) # URL starting with http
            s = re.sub(r'\w+?@.+?\\.com.*', URL_token,s)  # email
            s = re.sub(r'\[img.*?\]', IMG_token,s)  # image
            s = re.sub(r'< ?img.*?>', IMG_token, s)
            s = re.sub(r'@.*?(\s|$)', AT_token+'\\1', s)  # @id...
            s = re.sub('\u200B', '', s)
            s = s.strip()
            out.append(s)
        return out

    @staticmethod
    def traditional_to_simplified(sentences):
        '''繁体转简体'''
        return [HanziConv.toSimplified(s.strip()) for s in sentences]

    @staticmethod
    def process_for_segmented(sentences):
        '''数据处理,用于ltp分词'''
        out = []
        for s in sentences:
            s = re.sub(r'<@ID>|<URL>', '', s)
            s = re.sub(r'\s+', '，', s.strip())  # 将文本中的空白字符替换为逗号
            out.append(s)
        return out

    @staticmethod
    def removeQuotes(sentences):
        '''
        Remove punctuation from list of strings
        :param sentences: list with tokenised sentences
        :return: list
        '''
        out = []
        for s in sentences:
            out.append([w for w in s if not re.match(r"['`\"]+",w)])
            # # Twitter embeddings retain punctuation and use the following special tokens:
            # # <unknown>, <url>, <number>, <allcaps>, <pic>
            # # s = re.sub(r'[^\w\s]', ' ', s)
            # s = re.sub(r'[^a-zA-Z0-9_<>?.,]', ' ', s)
            # s = re.sub(r'[\s+]', ' ', s)
            # s = re.sub(r' +', ' ', s)  # prevent too much whitespace
            # s = s.lstrip().rstrip()
            # out.append(s)
        return out

    @staticmethod
    def stopwordslist(stopwords_file_path):
        '''加载停用词表'''
        stopwords = [line.strip() for line in open(stopwords_file_path, 'r', encoding='utf-8').readlines()]
        return stopwords

    @staticmethod
    def ltp_init():
        '''
        初始化ltp
        自定义词典 也可以使用
        ltp.init_dict(path="user_dict.txt", max_window=4)
        user_dict.txt 是词典文件， max_window是最大前向分词窗口 详见：https://ltp.ai/docs/quickstart.html#id6
        '''
        ltp = LTP()
        user_dict = ['新冠', '疫情', '90后', '00后']
        ltp.add_words(words=user_dict, max_window=4)
        return ltp

    @staticmethod
    def stopwordsList_en():
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.append('...')
        stopwords.append('___')
        stopwords.append('<url>')
        stopwords.append('<img>')
        stopwords.append('<URL>')
        stopwords.append('<IMG>')
        stopwords.append("can't")
        stopwords.append("i've")
        stopwords.append("i'll")
        stopwords.append("i'm")
        stopwords.append("that's")
        stopwords.append("n't")
        stopwords.append('rrb')
        stopwords.append('lrb')
        return stopwords

    @staticmethod
    def removeStopwords(sentence, stopwords_file_path):
        stopwords = Preprocessor.stopwordslist(stopwords_file_path)
        return [i for i in sentence if i not in stopwords]

    @staticmethod
    def removeShortLongWords(sentence):
        '''中文不可用'''
        return [w for w in sentence if len(w)>2 and len(w)<200]

    @staticmethod
    def tokenize_simple(iterator):
        return [sentence.split(' ') for sentence in iterator]

    @staticmethod
    def tokenize_nltk(iterator):
        return [nltk.word_tokenize(sentence) for sentence in iterator]

    @staticmethod
    def tokenize_tweet(iterator,strip=True):
        '''英文分词'''
        # tknzr = nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)
        tknzr = nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)
        result = [tknzr.tokenize(sentence) for sentence in iterator]
        if strip:
            result = [[w.replace(" ", "") for w in s] for s in result]
        return result

    @staticmethod
    def toLowerCase(sentences):
        out = []
        special_tokens = ['UNK','<IMG>','<URL>']
        for s in Preprocessor.tokenize_tweet(sentences):
            sent =[]
            # split sentences in tokens and lowercase except for special tokens
            for w in s:
                if w in special_tokens:
                    sent.append(w)
                else:
                    sent.append(w.lower())
            out.append(' '.join(sent))
        return out

    @staticmethod
    def max_document_length(sentences,tokenizer):
        sentences = tokenizer(sentences)
        return max([len(x) for x in sentences]) # tokenised length of sentence!

    @staticmethod
    def pad_sentences(sentences, max_length,pad_token='<PAD>',tokenized=False):
        '''
        Manually pad sentences with pad_token (to avoid the same representation for <unk> and <pad>)
        :param sentences:
        :param tokenizer:
        :param max_length:
        :param pad_token:
        :return:
        '''
        if tokenized:
            tokenized = sentences
            return [(s + [pad_token] * (max_length - len(s))) for s in tokenized]
        else:
            tokenized = Preprocessor.tokenize_tweet(sentences)
            return [' '.join(s + [pad_token] * (max_length - len(s))) for s in tokenized]

    @staticmethod
    def reduce_sentence_len(r_tok,max_len):
        '''
        Reduce length of tokenised sentence
        :param r_tok: nested list consisting of tokenised sentences e.g. [['w1','w2'],['w3']]
        :param max_len: maximum length of sentence
        :return: nested list consisting of tokenised sentences, none longer than max_len
        '''
        return [s if len(s) <= max_len else s[:max_len] for s in r_tok]

    @staticmethod
    def map_topics_to_id(r_tok,word2id_dict,s_max_len,opt):
        r_red = Preprocessor.reduce_sentence_len(r_tok, s_max_len)
        r_pad = Preprocessor.pad_sentences(r_red, s_max_len, pad_token='UNK', tokenized=True)
        mapped_sentences = []
        for s in r_pad:
            ids = [word2id_dict[lemma] if lemma in word2id_dict.keys() else 0 for lemma in s] # todo:fix 0 for UNK
            assert len(ids)==s_max_len, 'id len for {} should be {}, but is {}'.format(s,s_max_len,len(ids))
            mapped_sentences.append(np.array(ids))
        return np.array(mapped_sentences)

@cache
def tokenize_ltp(sentences, filepath='', postfix='cache'):
    '''
    ltp中文分词
    sentences: 要进行分词的句子列表  -> list
    sentence: 要进行分词的句子 ->str
    '''
    assert type(sentences) == list

    ltp = Preprocessor.ltp_init()  # 加载自定义词典
    result = []  # 分词结果
    for sentence in tqdm(sentences, desc='seg processing'):
        segment, _ = ltp.seg([sentence])
        result.append(segment[0])
    return result

