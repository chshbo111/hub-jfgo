#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类，并根据类内距离的排序输出
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return list(sentences)

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    all_sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(all_sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(all_sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    # 记录每个聚类所有词向量到中心的距离
    distance_label_dict = defaultdict(list)
    # 记录每个聚类所有句子
    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(all_sentences, kmeans.labels_):  #取出句子和标签
        #kmeans.cluester_centers_ #每个聚类中心
        distance = np.linalg.norm(kmeans.cluster_centers_[label]-vectors[all_sentences.index(sentence)]) 
        distance_label_dict[label].append(distance)
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    # 按照类内平均距离排序
    sorted_distances = sorted(
        distance_label_dict.items(), 
        key=lambda item : np.mean(item[1])
    )
    sorted_distances_dict = dict(sorted_distances)

    # 根据排序后的顺序将每个聚类的句子打印出来
    for label, distances in sorted_distances_dict.items():
        print("cluster %s, 平均类内距离 %s:" % (label, np.mean(distances)))
        for i in range(min(10, len(distances))):  #随便打印几个，太多了看不过来
            print(sentence_label_dict[label][i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()
    # distance_dict = {20: [1,3,4,5,2], 30: [3,4,6,7,6], 40: [4,6,3,1,6,7]}
    # for label, distances in distance_dict.items():
    #     print("label：%s, 个数 %s" % (label, len(distances)))
    #     print("距离列表：", distances)
    #     print("平均距离：", sum(distances)/len(distances))
    #     print("平均距离：", np.mean(distances))

    # sorted_items = sorted(
    #     distance_dict.items(), 
    #     key=lambda item : sum(item[1])/len(item[1])
    # )

    # print(dict(sorted_items))

    # p1 = np.array([1,2,3,4,5])
    # p2 = np.array([3,4,5,6,7])
    # print(cacl_distance(p1,p2))
    # print(np.linalg.norm(p1-p2))
    

