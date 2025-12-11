# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "../data/schema.json",
    "train_data_path": "../data/train.json",
    "valid_data_path": "../data/valid.json",
    "vocab_path":"../words.txt",
    "max_length": 15,
    "hidden_size": 256,
    "epoch": 20,
    "batch_size": 32,
    "epoch_data_size": 200,     #每轮训练中采样数量
    "optimizer": "adam",
    "learning_rate": 1e-3,
}