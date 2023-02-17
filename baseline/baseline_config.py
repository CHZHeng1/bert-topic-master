import os
import torch


class Config:
    """此类用于定义超参数"""
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.abspath(__file__))  # 获取Config类所在脚本的完整路径
        self.dataset_dir = os.path.join(self.project_dir, 'data')  # 数据集文件夹
        self.train_data_filepath = os.path.join(self.dataset_dir, 'train.csv')
        self.val_data_filepath = os.path.join(self.dataset_dir, 'test.csv')
        self.test_data_filepath = os.path.join(self.dataset_dir, 'test.csv')
        self.model_save_dir = os.path.join(self.project_dir, 'result')
        self.user_dict_filepath = os.path.join(self.dataset_dir, 'user_dict.txt')  # 自定义词典
        self.stopwords_filepath = os.path.join(self.dataset_dir, 'stopwords.txt')  # 停用词
        # 分词缓存文件
        self.segmented_data_save_dir = os.path.join(self.dataset_dir, 'segmented_cache')
        self.segmented_train_filepath = os.path.join(self.segmented_data_save_dir, 'segmented_train.txt')
        self.segmented_val_filepath = os.path.join(self.segmented_data_save_dir, 'segmented_test.txt')
        self.segmented_test_filepath = os.path.join(self.segmented_data_save_dir, 'segmented_test.txt')

        # 训练好的LDA文档主题分布
        self.upper_dir = os.path.dirname(self.project_dir)
        self.train_document_topics_filepath = \
            os.path.join(self.upper_dir, 'cache'+os.sep+'lda'+os.sep+'doc_topics_distribution'+os.sep+'train.npy')
        self.val_document_topics_filepath = \
            os.path.join(self.upper_dir, 'cache'+os.sep+'lda'+os.sep+'doc_topics_distribution'+os.sep+'test.npy')
        self.test_document_topics_filepath = \
            os.path.join(self.upper_dir, 'cache'+os.sep+'lda'+os.sep+'doc_topics_distribution'+os.sep+'test.npy')

        # 微博影响力和用户可信度
        self.train_manual_features_filepath = \
            os.path.join(self.upper_dir, 'cache'+os.sep+'data_cache'+os.sep+'manual_features_train.npy')
        self.val_manual_features_filepath = \
            os.path.join(self.upper_dir, 'cache' + os.sep + 'data_cache' + os.sep + 'manual_features_test.npy')
        self.test_manual_features_filepath = \
            os.path.join(self.upper_dir, 'cache'+os.sep+'data_cache'+os.sep+'manual_features_test.npy')

        # word2vec 模型保存位置
        self.word2vec_save_dir = os.path.join(self.model_save_dir, 'word2vec')

        self.tokenize = True  # 是否以分词的形式进行词典映射
        self.tokenize_type = 'ltp'  # ltp or jieba

        file_dir = [self.model_save_dir, self.segmented_data_save_dir]
        for dir_ in file_dir:
            if not os.path.exists(dir_):
                os.makedirs(dir_)

        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.batch_size = 128
        self.epochs = 10
        self.learning_rate = 1e-5

        self.embedding_dim = 300
        self.num_class = 2
        self.hidden_dim = 256

        self.topic_dim = 25  # lda主题个数
        self.manual_features_dim = 2
        # word2vec 独有参数
        self.vector_dim = 300  # word2vec生成词向量维度

        # TextCNN 独有参数
        self.filter_size = 3  # 卷积核宽度
        self.num_filter = 64  # 卷积核个数


if __name__ == '__main__':
    config = Config()


