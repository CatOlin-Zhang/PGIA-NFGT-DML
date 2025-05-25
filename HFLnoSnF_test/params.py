# params.py

import os

client_num_in_total = 8
client_num_per_round = 4

DATASET_SIZE_USED_TO_TRAIN = 6000  # 用于训练的数据集大小 最多 60000
DATASET_SIZE_USED_TO_EVAL = 1000  # 用于eval的数据集大小 最多 10000

MNIST_TRAIN_IMAGES_PATH = 'C:\\Capstone_project\\FED_ML\\MNIST\\train-images.idx3-ubyte'  # mnist 训练数据集路径
MNIST_TRAIN_LABELS_PATH = 'C:\\Capstone_project\\FED_ML\\MNIST\\train-labels.idx1-ubyte'  # mnist 训练标签路径
MNIST_EVAL_IMAGES_PATH = 'C:\\Capstone_project\\FED_ML\\MNIST\\t10k-images.idx3-ubyte'  # mnist eval数据集路径
MNIST_EVAL_LABELS_PATH = 'C:\\Capstone_project\\FED_ML\\MNIST\\t10k-labels.idx1-ubyte'  # mnist eval标签路径

UPLINK_BANDWIDTH = 50 * 1e3 * 1e3  # 上行链路带宽
CLIENT_TRANSMIT_POWER = 1  # 用户发送功率
NOISE_POWER_SPECTRAL_DENSITY = 10 ** (-10)  # 噪声功率谱密度
MODEL_SIZE = 1024 * 1024 * 3  # 模型大小
PATHLOSS_FACTOR = 2  # 路损因子



