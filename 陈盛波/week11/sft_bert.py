#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
import json
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

    def create_attention_mask(self, x, prompt_lengths):
        batch_size, sen_len = x.shape
        # 初始化全 0 矩阵
        mask = torch.zeros((batch_size, sen_len, sen_len), device=x.device)
        
        for i, p_len in enumerate(prompt_lengths):
            # 1. 对应图片中的 S1 部分：前 p_len 行全通，可以看到前 p_len 列
            mask[i, :p_len, :p_len] = 1
            
            # 2. 对应图片中的 S2 部分：
            if p_len < sen_len:
                # S2 看到左侧上下文 S1 (图片左下角白色区域)
                mask[i, p_len:, :p_len] = 1
                # S2 内部的自回归三角阵 (图片右下角白色梯形区域)
                causal_part = torch.tril(torch.ones((sen_len - p_len, sen_len - p_len), device=x.device))
                mask[i, p_len:, p_len:] = causal_part
                
        # 最终 mask 中 1 为可见，0 为灰色的不可见(遮蔽)部分
        return mask

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, promt_lengths, y=None):

        attention_mask = self.create_attention_mask(x, promt_lengths)

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
def build_sft_sample_with_len(prompt, answer, vocab, window_size):
    # 1. 编码 Prompt (S1 部分)
    # [CLS] + prompt_text + [SEP]
    prompt_tokens = [vocab.get(char, vocab["[UNK]"]) for char in prompt]
    # S1 定义为包含起始符和第一个分隔符的部分
    s1_ids = [vocab["[CLS]"]] + prompt_tokens + [vocab["[SEP]"]]
    prompt_len = len(s1_ids)

    # 2. 编码 Answer (S2 部分)
    # answer_text + [SEP]
    answer_tokens = [vocab.get(char, vocab["[UNK]"]) for char in answer]
    s2_ids = answer_tokens + [vocab["[SEP]"]]

    # 3. 拼接完整的输入 x
    input_ids = s1_ids + s2_ids
    
    # 4. 构造标签 y (Labels)
    # Prompt 部分(S1)全部设为 -100，Answer 部分(S2)保留原始 ID
    labels = [-100] * len(s1_ids) + s2_ids

    # 5. 截断处理
    if len(input_ids) > window_size:
        input_ids = input_ids[:window_size]
        labels = labels[:window_size]
    
    # 6. 填充处理 (Padding)
    # 输入补 0 ([PAD])，标签补 -100
    actual_length = len(input_ids)
    if actual_length < window_size:
        pad_len = window_size - actual_length
        input_ids += [vocab["[PAD]"]] * pad_len
        labels += [-100] * pad_len
        
    # 注意：prompt_len 不能超过 window_size
    prompt_len = min(prompt_len, window_size)

    return input_ids, labels, prompt_len

def load_data(corp_path):
    with open(corp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(batch_size, vocab, window_size, data):
    dataset_x = []
    dataset_y = []
    dataset_p_len = [] # 记录每个样本的 prompt 长度
    for item in random.sample(data, batch_size):
        x, y, promt_len = build_sft_sample_with_len(item['prompt'], item['answer'], vocab, window_size)
        dataset_x.append(x)
        dataset_y.append(y)
        dataset_p_len.append(promt_len)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y), torch.LongTensor(dataset_p_len)

#建立模型
def build_model():
    model = LanguageModel()
    return model

#测试代码
def generate_for_sft(prompt, model, vocab, window_size, max_gen_len=200):
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    
    # 1. 构造初始输入：[CLS] + prompt + [SEP]
    # 这部分对应的就是训练时的 S1 (Prefix)
    prompt_ids = [vocab.get(char, vocab["[UNK]"]) for char in prompt]
    input_ids = [vocab["[CLS]"]] + prompt_ids + [vocab["[SEP]"]]
    
    # 记录原始 Prompt 长度，用于最后截取答案
    prefix_len = len(input_ids)
    
    with torch.no_grad():
        for _ in range(max_gen_len):
            # 保持输入不超过 window_size
            curr_input = input_ids[-window_size:]
            x = torch.LongTensor([curr_input])
            
            # SFT 关键：传入当前 Prompt 的长度以构造正确的 Mask
            # 在推理时，随着生成增加，prompt_len 保持不变（即 Prefix 部分双向，后面单向）
            p_len = torch.LongTensor([min(prefix_len, len(curr_input))])
            
            if torch.cuda.is_available():
                x, p_len = x.cuda(), p_len.cuda()
            
            # 获取输出
            y_pred = model(x, p_len) # 调用带 p_len 的 forward
            
            # 取最后一个位置的概率分布进行采样
            last_token_prob = y_pred[0][-1]
            next_id = sampling_strategy(last_token_prob)
            
            # 如果生成了 [SEP] 或已达窗口上限，停止生成
            if next_id == vocab["[SEP]"] or len(input_ids) >= window_size + max_gen_len:
                break
                
            input_ids.append(next_id)
            
    # 2. 解码：只提取生成的 Answer 部分
    answer_ids = input_ids[prefix_len:]
    res = "".join([reverse_vocab.get(idx, "") for idx in answer_ids])
    return res

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
    window_size = 200       #样本文本长度
    vocab = build_vocab(Config["bert_path"])       #建立字表
    data = load_data(corpus_path)     #加载语料
    train_sample = len(data)   #每轮训练总共训练的样本总数
    model = build_model()    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=Config['learning_rate'])   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(Config['epoch']):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y, p_len = build_dataset(batch_size, vocab, window_size, data) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda(), p_len.cuba
            optim.zero_grad()    #梯度归零
            loss = model(x, p_len, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        prompts = ["感冒需要吃什么药？", "中国什么时候收复台湾？"]
        for prompt in prompts:
            print(prompt)
            print(generate_for_sft(prompt, model, vocab, window_size))
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
    train("corpus.json", False)
    
