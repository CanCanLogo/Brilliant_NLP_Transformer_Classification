import torch.nn as nn
import torch as torch
import math

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
    def __init__(self, vocab_size, ntoken, d_emb=512, d_hid=2048, nhead=5, nlayers=6, dropout=0.2, num_classes = 15, embedding_weight=None):
        super(Transformer_model, self).__init__()
        # 将"预训练的词向量"整理成 token->embedding 的二维映射矩阵 emdedding_weight 的形式，初始化 _weight
        # 当 emdedding_weight == None 时，表示随机初始化

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_emb, _weight=embedding_weight)

        self.pos_encoder = PositionalEncoding(d_model=d_emb, max_len=ntoken)
        self.encode_layer = nn.TransformerEncoderLayer(d_model=d_emb, nhead=nhead, dim_feedforward=d_hid)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encode_layer, num_layers=nlayers)


        #-----------------------------------------------------begin-----------------------------------------------------#
        # 请自行设计对 transformer 隐藏层数据的处理和选择方法
        self.dropout = nn.Dropout(dropout)  # 可选
        # self.decode_layer = nn.TransformerDecoderLayer(d_model=d_emb, nhead=nhead, dim_feedforward=d_hid)
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layer=self.decode_layer, num_layers=nlayers)
        # self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        # 请自行设计分类器

        # self.mlp = nn.Sequential(
        #     nn.Linear(ntoken * d_emb, d_emb * 9),
        #     nn.ReLU(True),
        #     nn.Linear(d_emb * 9, d_emb),
        #     nn.ReLU(True),
        #     nn.Linear(d_emb, 128),
        #     nn.ReLU(True),
        #     nn.Linear(128, 32),
        #     nn.ReLU(True),
        #     nn.Linear(32, num_classes),
        #     nn.ReLU(True)
        # )
        # self.fc = nn.Linear(d_emb, num_classes)
        self.fc = nn.Linear(d_emb * ntoken, num_classes)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)


        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x):
        x = self.embed(x)     
        x = x.permute(1, 0, 2)          
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)

        # x = self.transformer_decoder(x)
        # x = x.permute(1, 0, 2)
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 transformer_encoder 的隐藏层输出进行处理和选择，并完成分类
        x = self.dropout(x) # 可选
        # x = x.reshape(16, 80 * 300)
        x = self.flatten(x)
        # x = self.mlp(x)
        # torch.Size([16, 80, 15])
        # print(x.shape)

        # x = torch.max(x, dim=1)[0]  # 对隐藏层输出进行平均池化操作,将隐藏层输出的特征图从时间序列转换为固定长度的向量
        # x = torch.mean(x, dim=1)
        # x = torch.tensor(x, dtype=torch.float32)
        x = self.fc(x)
        # print(x.shape)
        # torch.Size([16, 15])
        # x = self.softmax(x)
        # x = self.fc(x)
        #------------------------------------------------------end------------------------------------------------------#
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

        self.classifier = nn.Linear(ntoken * d_hid * 2, num_classes)
        # 请自行设计分类器


        #------------------------------------------------------end------------------------------------------------------#

    def forward(self, x):
        x = self.embed(x)
        x = self.lstm(x)[0]
        #-----------------------------------------------------begin-----------------------------------------------------#
        # 对 bilstm 的隐藏层输出进行处理和选择，并完成分类
        x = self.dropout(x).reshape(-1, 50 * 80 * 2)   # ntoken*nhid*2 (2 means bidirectional)
        x = self.classifier(x)
        #------------------------------------------------------end------------------------------------------------------#
        return x