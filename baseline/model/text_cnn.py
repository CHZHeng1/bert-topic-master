from torch import nn
from torch.nn import functional as F


class TextCNN(nn.Module):
    """
    vocab_size: 词表大小
    embedding_dim: 经过embedding转换后词向量的维度
    filter_size: 卷积核的大小
    num_filter: 卷积核的个数
    num_class：类别数
    """
    def __init__(self, vocab_size, embedding_dim, filter_size, num_filter, num_class):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # padding=1 表示在卷积操作之前，将序列的前后各补充1个输入，这里没有找到详细的解释，考虑是为了让卷积核充分学习序列的信息
        self.conv1d = nn.Conv1d(embedding_dim, num_filter, filter_size, padding=1)
        self.activate = F.relu
        self.linear = nn.Linear(num_filter, num_class)

    def forward(self, inputs):
        # 输入数据维度 （batch_size, max_src_len）
        embedding = self.embedding(inputs)
        # 经过embedding层后，维度变为（batch_size, max_src_len, embedding_dim）
        # 但卷积层要求输入数据的形状为（batch_size, in_channels, max_src_len）,因此这里对输入数据进行维度交换
        # embedding.permute(0,2,1) 交换embedding中维度2和1的位置
        convolution = self.activate(self.conv1d(embedding.permute(0, 2, 1)))
        # 经过卷积层输出以后的维度为（batch_size, out_channels, out_seq_len）
        pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2])  # 池化层聚合
        # 经过池化层输出以后的维度为（batch_size,out_channels,1）
        # 但由于全连接层要求输入数据的最后一个维度为卷积核的个数，即out_channels,因此这里需要降维
        probs = self.linear(pooling.squeeze(dim=2))
        return probs