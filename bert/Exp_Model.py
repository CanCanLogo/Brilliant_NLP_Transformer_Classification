import torch.nn as nn
import torch as torch
import math
from transformers import BertModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer_model(nn.Module):
    def __init__(self, ntoken, d_emb=768, dropout=0.1, num_classes = 15):
        super(Transformer_model, self).__init__()
        self.bert = BertModel.from_pretrained('../bert-base-chinese')
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_emb, num_classes)
        self.relu = nn.ReLU()
    def forward(self ,input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id,
                                     attention_mask=mask,
                                     return_dict=False)
        # 注意我们取用bert模型输出的第二个,也即[CLS]来用于后续分类
        x = self.dropout(pooled_output) # 生成维度：(1，batch, d_emb)
        x = self.fc(x)
        x = self.relu(x)
        return x
    
    
class BiLSTM_model(nn.Module):
    def __init__(self, vocab_size, ntoken, d_emb=100, d_hid=80, nlayers=1, dropout=0.2, num_classes = 16, embedding_weight=None):
        super(BiLSTM_model, self).__init__()
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 当 emdedding_weight == None 时，表示随机初始化
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)

        self.lstm = nn.LSTM(input_size=d_emb, hidden_size=d_hid, num_layers=nlayers, bidirectional=True, batch_first=True)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 bilstm 隐藏层数据的处理和选择方法
        self.dropout = nn.Dropout(dropout)  # 可选

        self.classifier = nn.Linear(50 * d_hid * 2, num_classes)
        # 请自行设计分类器


        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x):
        x = self.embed(x)
        x = self.lstm(x)[0]
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 bilstm 的隐藏层输出进行处理和选择，并完成分类
        x = self.dropout(x).reshape(-1, 50*80*2)   # ntoken*nhid*2 (2 means bidirectional)
        x = self.classifier(x)
        #------------------------------------------------------end------------------------------------------------------#
        return x