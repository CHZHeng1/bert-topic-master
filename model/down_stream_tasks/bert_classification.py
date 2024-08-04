from model.bert_model.bert import BertModel
import torch
import torch.nn as nn


# BERT
class BertForSentenceClassification(nn.Module):
    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertForSentenceClassification, self).__init__()
        self.num_labels = config.num_labels
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:  # 返回一个随机初始化参数的BERT模型
            self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len]
                token_type_ids=None,  # [src_len, batch_size] 单句分类时为None
                position_ids=None,  # [1,src_len]
                labels=None):  # [batch_size,]
        pooled_output, _ = self.bert.forward(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             token_type_ids=token_type_ids,
                                             position_ids=position_ids)  # [batch_size,hidden_size]
        # pooled_output为BERT第一个位置的向量经过一个全连接层后的结果，第二个参数是BERT中所有位置的向量
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_label]
        if labels is not None:
            # 多分类
            loss_fct = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数要求输入标签类型为long
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # 二分类
            # loss_fct = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失函数要求输入标签类型为float
            # loss = loss_fct(logits.view(-1, 1), labels.float().view(-1, 1))  # 计算准确率时需要设置阈值
            return loss, logits
        else:
            return logits


# 辅助函数，用于将连续型数值转换为分类型数值
def cal(zhat, p=0.5):
    sigma = torch.sigmoid(zhat)
    return ((sigma >=p).float())


# 计算准确率
def accuracy(zhat, y):
    acc_bool = cal(zhat).flatten() == y.flatten()
    acc = torch.mean(acc_bool.float())
    return acc


# BERT + document_topics
class BertWithTopics(nn.Module):
    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertWithTopics, self).__init__()
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size + config.num_topics
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:  # 返回一个随机初始化参数的BERT模型
            self.bert = BertModel(config)
        self.full_connection = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len]
                token_type_ids=None,  # [src_len, batch_size] 单句分类时为None
                position_ids=None,  # [1,src_len]
                labels=None,  # [batch_size,]
                topic_vector=None):  # [batch_size, num_topics]
        bert_output, _ = self.bert(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids)
        bert_output = self.dropout(bert_output)

        pooled_output = torch.cat([bert_output, topic_vector], dim=1)
        pooled_output = self.full_connection(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_label]
        if labels is not None:
            # 多分类
            loss_fct = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数要求输入标签类型为long
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # 二分类
            # loss_fct = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失函数要求输入标签类型为float
            # loss = loss_fct(logits.view(-1, 1), labels.float().view(-1, 1))  # 计算准确率时需要设置阈值
            return loss, logits
        else:
            return logits


# BERT + word topic
class BertWithWordTopics(nn.Module):
    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertWithWordTopics, self).__init__()
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size + config.num_topics
        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:  # 返回一个随机初始化参数的BERT模型
            self.bert = BertModel(config)
        self.full_connection = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len]
                token_type_ids=None,  # [src_len, batch_size] 单句分类时为None
                position_ids=None,  # [1,src_len]
                labels=None,  # [batch_size,]
                word_topic_ids=None,  # [batch_size, max_src_len]
                word_topic_length=None,  # 序列, batch_size个样本长度
                topic_matrix=None):
        bert_output, _ = self.bert(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids)
        bert_output = self.dropout(bert_output)

        # [max_src_len, topic_num]
        word_topics_tensor = [torch.index_select(topic_matrix, dim=0, index=word_topic_ids[ind]) for ind in
                              range(word_topic_ids.shape[0])]
        word_topics_tensor = torch.stack(word_topics_tensor)  # [batch_size, max_src_len, topic_nums]

        # topic_vector = torch.sum(word_topics_tensor, dim=1).transpose(0, 1)  # [topic_nums, batch_size]
        # length_diag = torch.diag(torch.tensor(1.) / word_topic_length)  # [batch_size, batch_size]
        # topic_vector = torch.mm(topic_vector, length_diag).transpose(0, 1)  # [batch_size, topic_nums]

        topic_vector = torch.mean(word_topics_tensor, dim=1)

        pooled_output = torch.cat([bert_output, topic_vector], dim=1)
        pooled_output = self.full_connection(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_label]
        if labels is not None:
            # 多分类
            loss_fct = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数要求输入标签类型为long
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # 二分类
            # loss_fct = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失函数要求输入标签类型为float
            # loss = loss_fct(logits.view(-1, 1), labels.float().view(-1, 1))  # 计算准确率时需要设置阈值
            return loss, logits
        else:
            return logits


# BERT + document topic + weibo influence + user reliability
class BertWithDocumentWeiboUser(nn.Module):
    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertWithDocumentWeiboUser, self).__init__()
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size + config.num_topics + config.manual_features_size

        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:  # 返回一个随机初始化参数的BERT模型
            self.bert = BertModel(config)

        self.full_connection = nn.Linear(self.hidden_size, self.hidden_size)
        # self.ln_layer = nn.LayerNorm(self.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len]
                token_type_ids=None,  # [src_len, batch_size] 单句分类时为None
                position_ids=None,  # [1,src_len]
                labels=None,  # [batch_size,]
                topic_vector=None,  # [batch_size, num_topics]
                feature_variable=None
                ):
        bert_output, _ = self.bert(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids)
        bert_output = self.dropout(bert_output)

        feature_cat = torch.cat([bert_output, topic_vector, feature_variable], dim=1)
        fc_layer_out = self.full_connection(feature_cat)
        # ln_layer_out = self.ln_layer(fc_layer_out)
        pooled_output = self.activation(fc_layer_out)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_label]
        if labels is not None:
            # 多分类
            loss_fct = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数要求输入标签类型为long
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # 二分类
            # loss_fct = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失函数要求输入标签类型为float
            # loss = loss_fct(logits.view(-1, 1), labels.float().view(-1, 1))  # 计算准确率时需要设置阈值
            return loss, logits
        else:
            return logits


# BERT + word topic + weibo influence + user reliability
class BertWithWordWeiboUser(nn.Module):
    def __init__(self, config, bert_pretrained_model_dir=None):
        super(BertWithWordWeiboUser, self).__init__()
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size + config.num_topics + config.manual_features_size

        if bert_pretrained_model_dir is not None:
            self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
        else:  # 返回一个随机初始化参数的BERT模型
            self.bert = BertModel(config)

        self.full_connection = nn.Linear(self.hidden_size, self.hidden_size)
        # self.ln_layer = nn.LayerNorm(self.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids,  # [src_len, batch_size]
                attention_mask=None,  # [batch_size, src_len]
                token_type_ids=None,  # [src_len, batch_size] 单句分类时为None
                position_ids=None,  # [1,src_len]
                labels=None,  # [batch_size,]
                word_topic_ids=None,  # [batch_size, max_src_len]
                feature_variable=None,
                topic_matrix=None):
        bert_output, _ = self.bert(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids)
        bert_output = self.dropout(bert_output)

        # [max_src_len, topic_num]
        word_topics_tensor = [torch.index_select(topic_matrix, dim=0, index=word_topic_ids[ind]) for ind in
                              range(word_topic_ids.shape[0])]
        word_topics_tensor = torch.stack(word_topics_tensor)  # [batch_size, max_src_len, topic_nums]
        topic_vector = torch.mean(word_topics_tensor, dim=1)

        pooled_output = torch.cat([bert_output, topic_vector, feature_variable], dim=1)
        pooled_output = self.full_connection(pooled_output)
        # pooled_output = self.ln_layer(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_label]
        if labels is not None:
            # 多分类
            loss_fct = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数要求输入标签类型为long
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # 二分类
            # loss_fct = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失函数要求输入标签类型为float
            # loss = loss_fct(logits.view(-1, 1), labels.float().view(-1, 1))  # 计算准确率时需要设置阈值
            return loss, logits
        else:
            return logits


class BertTopic(nn.Module):
    def __init__(self, config, feature_name=None, bert_pretrained_model_dir=None):
        super(BertTopic, self).__init__()
        if feature_name == 'BERT-W-I' or feature_name == 'BERT-W-R' or feature_name == 'BERT-I-R':
            if bert_pretrained_model_dir is not None:
                self.bert = BertModel.from_pretrained(config, bert_pretrained_model_dir)
            else:  # 返回一个随机初始化参数的BERT模型
                self.bert = BertModel(config)
            if feature_name == 'BERT-W-I' or feature_name == 'BERT-W-R':
                self.hidden_size = config.hidden_size + config.num_topics + 1
            elif feature_name == 'BERT-I-R':
                self.hidden_size = config.hidden_size + config.manual_features_size
        elif feature_name == 'W-I-R':
            self.hidden_size = config.num_topics + config.manual_features_size
            # self.hidden_size = config.num_topics

        self.num_labels = config.num_labels
        self.full_connection = nn.Linear(self.hidden_size, self.hidden_size)
        # self.ln_layer = nn.LayerNorm(self.hidden_size)
        self.activation = torch.tanh
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, labels=None,
                word_topics_id=None, word_topics_embedding=None, feature_influence=None, feature_reliability=None):
        features_concat = None
        if input_ids is not None:
            bert_output, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                       token_type_ids=token_type_ids, position_ids=position_ids)
            bert_output = self.dropout(bert_output)
            if word_topics_id is not None and feature_influence is not None:
                word_topics_tensor = [torch.index_select(word_topics_embedding, dim=0, index=word_topics_id[ind]) for
                                      ind in range(word_topics_id.shape[0])]
                word_topics_tensor = torch.stack(word_topics_tensor)  # [batch_size, max_src_len, topic_nums]
                topic_vector = torch.mean(word_topics_tensor, dim=1)
                features_concat = torch.cat([bert_output, topic_vector, feature_influence], dim=1)

            elif word_topics_id is not None and feature_reliability is not None:
                word_topics_tensor = [torch.index_select(word_topics_embedding, dim=0, index=word_topics_id[ind]) for
                                      ind in range(word_topics_id.shape[0])]
                word_topics_tensor = torch.stack(word_topics_tensor)  # [batch_size, max_src_len, topic_nums]
                topic_vector = torch.mean(word_topics_tensor, dim=1)
                features_concat = torch.cat([bert_output, topic_vector, feature_reliability], dim=1)

            elif feature_influence is not None and feature_reliability is not None:
                features_concat = torch.cat([bert_output, feature_influence, feature_reliability], dim=1)

        elif word_topics_id is not None and feature_influence is not None and feature_reliability is not None:
            word_topics_tensor = [torch.index_select(word_topics_embedding, dim=0, index=word_topics_id[ind]) for
                                  ind in range(word_topics_id.shape[0])]
            word_topics_tensor = torch.stack(word_topics_tensor)  # [batch_size, max_src_len, topic_nums]
            topic_vector = torch.mean(word_topics_tensor, dim=1)
            features_concat = torch.cat([topic_vector, feature_influence, feature_reliability], dim=1)
            # features_concat = topic_vector

        pooled_output = self.full_connection(features_concat)
        # pooled_output = self.ln_layer(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_label]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits








# if __name__ == '__main__':
#     pass




