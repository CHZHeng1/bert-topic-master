import os
import logging
import torch
from model.bert_model.bert_config import BertConfig
from utils.log_helpers import logger_init


class ModelConfig:
    def __init__(self):
        # 获取当前执行脚本的完整路径
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        # print(self.project_dir)
        self.dataset_dir = os.path.join(self.project_dir, 'data')  # 存放数据集文件夹
        self.pretrained_model_dir = os.path.join(self.project_dir, "bert_base_chinese")
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        self.train_file_path = os.path.join(self.dataset_dir, 'train.csv')
        self.val_file_path = os.path.join(self.dataset_dir, 'test.csv')
        self.test_file_path = os.path.join(self.dataset_dir, 'test.csv')
        self.stopwords_file_path = os.path.join(self.dataset_dir, 'stopwords.txt')
        self.user_dict_file_path = os.path.join(self.dataset_dir, 'user_dict.txt')

        self.fake_news_dir = os.path.join(self.dataset_dir, 'fake_news')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')  # 模型存放文件夹
        self.data_save_dir = os.path.join(self.model_save_dir, 'data_cache')  # 数据集缓存文件存放文件夹
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.results_save_dir = os.path.join(self.project_dir, 'results')

        # 分词缓存
        self.segmented_train_file_path = os.path.join(self.data_save_dir,
                                                      'segmented_cache'+os.sep+'segmented_train.txt')
        self.segmented_val_file_path = os.path.join(self.data_save_dir,
                                                    'segmented_cache' + os.sep + 'segmented_test.txt')
        self.segmented_test_file_path = os.path.join(self.data_save_dir,
                                                     'segmented_cache'+os.sep+'segmented_test.txt')
        # 分词工具
        self.tokenize_type = 'ltp'

        # document topic
        self.train_document_topics_file_path = os.path.join(self.model_save_dir,
                                                            'lda'+os.sep+'doc_topics_distribution'+os.sep+'train.npy')
        self.test_document_topics_file_path = os.path.join(self.model_save_dir,
                                                           'lda'+os.sep+'doc_topics_distribution'+os.sep+'test.npy')
        # word topic
        self.word_topics_file_path = os.path.join(self.model_save_dir,
                                                  'lda'+os.sep+'word_topics_distribution'+os.sep+'word_topics.log')
        # manual features
        self.train_manual_features_file_path = os.path.join(self.data_save_dir, 'manual_features_train.npy')
        self.val_manual_features_file_path = os.path.join(self.data_save_dir, 'manual_features_test.npy')
        self.test_manual_features_file_path = os.path.join(self.data_save_dir, 'manual_features_test.npy')

        self.train_old_influence_file_path = os.path.join(self.data_save_dir, 'old_influence_train.npy')
        self.val_old_influence_file_path = os.path.join(self.data_save_dir, 'old_influence_test.npy')
        self.test_old_influence_file_path = os.path.join(self.data_save_dir, 'old_influence_test.npy')

        self.unk_topic = 'zero'  # 未登录词的处理方式  zero or uniform

        self.split_sep = '\t'
        self.is_sample_shuffle = True
        self.batch_size = 40
        # self.learning_rate = 5e-5
        self.epochs = 10

        self.max_sen_len = None
        self.num_labels = 2
        self.model_val_per_epoch = 2

        self.max_position_embeddings = 512
        self.pad_token_id = 0
        self.num_topics = 25  # 主题个数
        self.manual_features_size = 2

        # 使用torch框架中的多头注意力机制模块
        self.use_torch_multi_head = True

        logger_init(log_file_name='bert_topic', log_level=logging.INFO,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # 把原始bert中的配置参数也导入进来
        bert_config_path = os.path.join(self.pretrained_model_dir, "config.json")
        bert_config = BertConfig.from_json_file(bert_config_path)
        for key, value in bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"###  {key} = {value}")


if __name__ == '__main__':
    config = ModelConfig()
    # print(config.train_document_topics_file_path)
