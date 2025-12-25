#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertTokenizer
from config import Config
from datetime import datetime
"""
基于pytorch的Bert语言模型
"""

class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()
        self.layer = BertModel.from_pretrained(Config["bert_path"], return_dict=False)
        self.classify = nn.Linear(self.layer.config.hidden_size, self.layer.config.vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def create_attention_mask(self, x):
        batch_size, sen_len = x.shape
        
        # 1. 依然构造下三角矩阵
        mask = torch.tril(torch.ones((sen_len, sen_len), device=x.device))
        
        # 2. 转换为 BERT 扩展格式
        # 关键：手动调用模型内部的转换函数，它会自动处理维度偏移和 -10000.0
        # 注意：这里传进去的必须是 2D 的 0/1 矩阵，但因为我们要下三角，
        # 所以我们需要欺骗一下模型，传一个 [batch, seq, seq] 进去
        
        # 构造 [batch, seq, seq] 的 0/1 矩阵
        causal_mask = mask.unsqueeze(0).repeat(batch_size, 1, 1) # [64, 10, 10]

        return causal_mask

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):

        attention_mask = self.create_attention_mask(x)

        # 传入Mask
        x, _ = self.layer(input_ids=x, attention_mask=attention_mask)

        # x, _ = self.layer(input_ids=x, attention_mask=mask.float())        #output shape:(batch_size, sen_len, input_dim)
        y_pred = self.classify(x)   #output shape:(batch_size, sen_len, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

#加载字表
def build_vocab(vocab_path):
    return BertTokenizer.from_pretrained(vocab_path).vocab

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample(window_size, vocab, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位
    x = [vocab.get(word, vocab["[UNK]"]) for word in window]   #将字转换成序号
    y = [vocab.get(word, vocab["[UNK]"]) for word in target]
    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(window_size, vocab, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model():
    model = LanguageModel()
    return model

#文本生成测试代码
def generate_sentence(openings, model, vocab, window_size):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char
            x = [vocab.get(char, vocab["[UNK]"]) for char in openings[-window_size:]]
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]
    return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


#计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["[UNK]"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["[UNK]"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, save_weight=True):
    batch_size = Config['batch_size']       #每次训练样本个数
    train_sample = 50000   #每轮训练总共训练的样本总数
    window_size = 10       #样本文本长度
    vocab = build_vocab(Config["bert_path"])       #建立字表
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model()    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=Config['learning_rate'])   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(Config['epoch']):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size))
        print(generate_sentence("我之前从山中来，带着兰花草", model, vocab, window_size))
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train("corpus.txt", False)
    
