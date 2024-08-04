import os
from config import ModelConfig


class TopicConfig(ModelConfig):
    def __init__(self):
        super(TopicConfig, self).__init__()

        self.lda_model_save_dir = os.path.join(self.model_save_dir, 'lda')  # 模型存放位置
        self.segmented_data_save_dir = os.path.join(self.data_save_dir, 'segmented_cache')  # 分词数据缓存文件夹
        self.segmented_file_path = os.path.join(self.segmented_data_save_dir, 'segmented_cache.txt')
        self.segmented_train_file_path = os.path.join(self.segmented_data_save_dir, 'segmented_train.txt')
        self.segmented_val_file_path = os.path.join(self.segmented_data_save_dir, 'segmented_test.txt')
        self.segmented_test_file_path = os.path.join(self.segmented_data_save_dir, 'segmented_test.txt')
        self.processed_file_path = os.path.join(self.segmented_data_save_dir, 'processed_text.txt')
        self.stopwords_file_path = os.path.join(self.dataset_dir, 'stopwords.txt')
        self.user_dict_file_path = os.path.join(self.dataset_dir, 'user_dict.txt')
        self.topic_results_save_dir = os.path.join(self.results_save_dir, 'topic_model')
        self.lda_results_save_dir = os.path.join(self.topic_results_save_dir, 'lda')
        self.lda_doc_topics_distribution_save_dir = os.path.join(self.lda_model_save_dir, 'doc_topics_distribution')

        self.train_document_topics_file_path = os.path.join(self.lda_doc_topics_distribution_save_dir, 'train.npy')
        self.test_document_topics_file_path = os.path.join(self.lda_doc_topics_distribution_save_dir, 'test.npy')
        self.val_document_topics_file_path = os.path.join(self.lda_doc_topics_distribution_save_dir, 'test.npy')

        self.lda_word_topics_distribution_save_dir = os.path.join(self.lda_model_save_dir, 'word_topics_distribution')

        self.epochs = 10
        self.learning_rate = 0.07
        self.batch_size = 40
        self.tokenize_type = 'ltp'  # ltp or jieba
        self.num_topics = 25  # lda主题个数
        self.topic_type = 'LDA'
        self.unk_topic = 'zero'  # 未登录词的处理方式  zero or uniform
        self.topic_scope = ['doc', 'word']  # ['', 'word', 'doc', 'word+doc', 'word+avg']
        # self.recover_topic_peaks = False  # True or False


if __name__ == '__main__':
    topic_model_config = TopicConfig()
    print(topic_model_config.lda_model_save_dir)
