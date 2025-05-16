import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset
import jieba

from gensim.models.keyedvectors import KeyedVectors

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese')

# stopwords = []
    # stopwords = mystrip(stopwords)
with open("../../stopwords.txt", encoding="utf-8") as f:
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

class Data(torch.utils.data.Dataset):
    def __init__(self, path, max_token_per_sent, test_mode=False):
        self.dictionary = Dictionary('../tnews/')
        self.max_token_per_sent = max_token_per_sent
        self.texts, self.labels = self.tokenize(path, test_mode)

    def tokenize(self, path, test_mode=False):
        sentences = []
        labels = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                one_data = json.loads(line)  # 读取一条数据
                sent = one_data['sentence']
                sentences.append(sent)  # 将原句存储在列表sentences中
                if test_mode:
                    label = json.loads(line)['id']
                    labels.append(label) # 测试集无标签，在 label 中存测试数据的 id
                else:
                    label = json.loads(line)['label']
                    labels.append(self.dictionary.label2idx[label])
        idss = [tokenizer(text,
                        padding='max_length',
                        max_length = self.max_token_per_sent,
                        truncation=True,  # 所有句子都被截断或填充到相同的长度
                        return_tensors="pt")  # 返回PyTorch张量
               for text in sentences]
        return idss, labels

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y

class Corpus(object):
    '''
    完成对数据集的读取和预处理，处理后得到所有文本数据的对应的 token 表示及相应的标签。
    
    该类适用于任务一、任务二，若要完成任务三，需对整个类进行调整，例如，可直接调用预训练模型提供的 tokenizer 将文本转为对应的 token 序列。
    '''
    def __init__(self, path, max_token_per_sent):
        self.dictionary = Dictionary(path)

        self.max_token_per_sent = max_token_per_sent

        self.train, _= self.tokenize(os.path.join(path, 'train.json'))
        self.valid = self.tokenize(os.path.join(path, 'dev.json'))
        self.test = self.tokenize(os.path.join(path, 'test.json'), True)


    def tokenize(self, path, test_mode=False):
        '''
        处理指定的数据集分割，处理后每条数据中的 sentence 都将转化成对应的 token 序列。
        '''
        idss = []
        labels = []
        mask = []
        with open(path, 'r', encoding='utf8') as f:
            for line in f:
                one_data = json.loads(line)  # 读取一条数据
                sent = one_data['sentence']
                idss.append(sent)
                # jieba.setLogLevel(jieba.logging.INFO)
                # sent = jieba.lcut(sent, cut_all=True)
                # sent = self.remove_stopwords(sent)
                # ids = [tokenizer(sent,
                #                 padding='max_length',
                #                 max_length = self.max_token_per_sent,
                #                 truncation=True,
                #                 return_tensors="pt")]
                # ids = list(ids[0].values())
                # print(ids)
                # idss.append(ids)
                # idss.append(ids['input_ids'])
                # mask.append(ids['attention_mask'])
                '''
                [{'input_ids': tensor([[ 101, 1343, 3805, 1744, 8024, 5632, 4507, 6121, 1962, 6820, 3221, 6656,
         1730, 3683, 6772, 1962, 8043,  102,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0]])}]
                '''

                # 测试集无标签，在 label 中存测试数据的 id，便于最终预测文件的打印
                if test_mode:
                    label = json.loads(line)['id']      
                    labels.append(label)
                else:
                    label = json.loads(line)['label']
                    labels.append(self.dictionary.label2idx[label])
            # idss = idss.values
            # idss = torch.tensor(idss)
            # mask = torch.tensor(mask)
            # labels = torch.tensor(np.array(labels)).long()
            # print('labels')
            # print(len(labels))

        idss = [tokenizer(text,
                        padding='max_length',
                        max_length = self.max_token_per_sent,
                        truncation=True,
                        return_tensors="pt")
               for text in idss
               ]
        print(idss)
        # return TensorDataset(idss, mask, labels)
        return (idss, labels)

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

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y