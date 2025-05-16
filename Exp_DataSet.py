import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset
import jieba

from gensim.models.keyedvectors import KeyedVectors

# stopwords = []
    # stopwords = mystrip(stopwords)
with open("../stopwords.txt", encoding="utf-8") as f:
    stopwords = f.readlines()
    for i in range(len(stopwords)):
        stopwords[i] = stopwords[i].strip("\n")

class Dictionary(object):
    def __init__(self, path):

        self.word2tkn = {"[PAD]": 0}
        self.tkn2word = ["[PAD]"]

        self.label2idx = {}
        self.idx2label = []

        # 获取 label 的 映射
        with open(os.path.join(path, 'labels.json'), 'r', encoding='utf-8') as f:
            for line in f:
                one_data = json.loads(line)
                label, label_desc = one_data['label'], one_data['label_desc']
                self.idx2label.append([label, label_desc])
                self.label2idx[label] = len(self.idx2label) - 1

    def add_word(self, word):
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]


class Corpus(object):
    '''
    完成对数据集的读取和预处理，处理后得到所有文本数据的对应的 token 表示及相应的标签。
    
    该类适用于任务一、任务二，若要完成任务三，需对整个类进行调整，例如，可直接调用预训练模型提供的 tokenizer 将文本转为对应的 token 序列。
    '''
    def __init__(self, path, max_token_per_sent, emb_path):
        self.dictionary = Dictionary(path)

        self.max_token_per_sent = max_token_per_sent

        self.train = self.tokenize(os.path.join(path, 'train.json'))
        self.valid = self.tokenize(os.path.join(path, 'dev.json'))
        self.test = self.tokenize(os.path.join(path, 'test.json'), True)



        #-----------------------------------------------------begin-----------------------------------------------------#
        # 若要采用预训练的 embedding, 需处理得到 token->embedding 的映射矩阵 embedding_weight。矩阵的格式参考 nn.Embedding() 中的参数 _weight
        # 注意，需考虑 [PAD] 和 [UNK] 两个特殊词向量的设置

        self.embedding_weight = self.dic2embedding(self.dictionary.tkn2word, emb_path)
        self.embedding_weight = torch.cat([tensor.unsqueeze(0) for tensor in self.embedding_weight], dim=0)

        self.stopwords = []

        #------------------------------------------------------end------------------------------------------------------#
    def dic2embedding(self, dictionary, emb_path, WORD_DIM = 300):
        # 从整型到vector构建一个向量表，行数指的是字典word2tkn的数，每个词对应的数，每一行是一个该词对应的词向量
        word_vectors = KeyedVectors.load_word2vec_format(emb_path)
        # print("loading end")
        embedding_weight = []
        pad_vector = np.zeros(WORD_DIM, dtype=np.float32)
        embedding_weight.append(pad_vector)
        UNK_num = 0
        for i in range(1, len(dictionary)):
            word = dictionary[i]
            if word in word_vectors:  # 如果是登录词 得到其词向量表示
                vector = word_vectors.get_vector(word)
                vector = vector.astype(np.float32)
                # print(vector)
            else:  # 如果不是登录词 设置为随机词向量,UNK情况
                vector = np.random.uniform(-0.01, 0.01, WORD_DIM).astype("float32")
                UNK_num += 1
            embedding_weight.append(vector)
        print('UNK率')
        print(UNK_num / len(dictionary))
        embedding_weight_torch = torch.tensor(np.array(embedding_weight, dtype=np.float32), dtype=torch.float32)
        return embedding_weight_torch

    '''
        默认使用单精度float32训练模型RuntimeError: expected scalar type Double but found Float
        '''


    def pad(self, origin_token_seq):
        '''
        padding: 将原始的 token 序列补 0 至预设的最大长度 self.max_token_per_sent
        '''
        if len(origin_token_seq) > self.max_token_per_sent:
            return origin_token_seq[:self.max_token_per_sent]
        else:
            return origin_token_seq + [0 for _ in range(self.max_token_per_sent-len(origin_token_seq))]

    def tokenize(self, path, test_mode=False):
        '''
        处理指定的数据集分割，处理后每条数据中的 sentence 都将转化成对应的 token 序列。
        '''
        idss = []
        labels = []

        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                one_data = json.loads(line)  # 读取一条数据
                sent = one_data['sentence']
                # print(sent)
                '''
                原句子
                '''
                #-----------------------------------------------------begin-----------------------------------------------------#
                # 若要采用预训练的 embedding, 需在此处对 sent 进行分词
                jieba.setLogLevel(jieba.logging.INFO)
                sent = jieba.lcut(sent, cut_all=True)
                # pos_sentence.append(remove_stopwords(words))
                sent = self.remove_stopwords(sent)
                #------------------------------------------------------end------------------------------------------------------#
                # 向词典中添加词
                for word in sent:
                    # print(word)
                    # 单字
                    self.dictionary.add_word(word)

                ids = []
                for word in sent:
                    # print(self.dictionary.word2tkn[word])
                    '数字'
                    ids.append(self.dictionary.word2tkn[word])
                idss.append(self.pad(ids))
                
                # 测试集无标签，在 label 中存测试数据的 id，便于最终预测文件的打印
                if test_mode:
                    label = json.loads(line)['id']      
                    labels.append(label)
                else:
                    label = json.loads(line)['label']
                    labels.append(self.dictionary.label2idx[label])

            idss = torch.tensor(np.array(idss))
            labels = torch.tensor(np.array(labels)).long()
            # print('labels')
            # print(len(labels))
            '''
            labels
            tensor([7, 4, 5,  ..., 7, 6, 6])
            labels
            tensor([ 2,  9,  4,  ..., 13,  8, 11])
            labels
            tensor([   0,    1,    2,  ..., 9997, 9998, 9999])
            labels
            53360
            labels
            10000
            labels
            10000
            '''

            
        return TensorDataset(idss, labels)


    #
    def remove_stopwords(self, _words):
        """
        函数功能：去掉为空的分词

        """
        global stopwords
        _i = 0
        for _ in range(len(_words)):
            if _words[_i] in stopwords or _words[_i].strip() == "":
                # print(_words[_i])
                _words.pop(_i)
            else:
                _i += 1
        return _words