# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "model_type":"bert",
    "max_length": 100,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 128,
    "tuning_tactics":"lora_tuning",
    # "tuning_tactics":"finetuing",
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": True,
    "pretrain_model_path":r"D:\nlpcodes\bert-base-chinese",
    "seed": 987
}