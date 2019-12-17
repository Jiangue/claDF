from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras.layers import Input, Dense, concatenate, Flatten, Dropout, LeakyReLU, BatchNormalization
from keras import Sequential
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.externals import joblib

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import os

import math
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import h5py

# 定义一个Tree_down方法类
class tree_down(object):

    # 定义一个数据格式转化的函数
    def trasform_data_format(data, label):
        r, c = data.shape

        # 创建空的多维数组用于存放数据
        dataset_array = np.zeros(shape = (c, r, 1))
        # 创建空的数组用于存放图片的标注信息
        dataset_labels = np.zeros(shape=(c), dtype=np.uint8)
        # 从文件夹下读取数据
        index = 0
        for i in range(c):
            data_reshaped = np.reshape(data[:, i], newshape=(1, r, 1))
            # 将维度转换后的图片存入指定的数组内
            dataset_array[index, :, :] = data_reshaped
            dataset_labels[index] = label[i]
            index = index + 1
        return dataset_array, dataset_labels

    def Perceptron( ):

        model = Sequential()
        model.add(Flatten(input_shape=(1500, 1)))
        model.add(Dense(3, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        # 返回编译完成的model
        return model

    def key_gene_set(gene, weight_matrix):
        """
        :param gene: 基因列表
        :param weight_matrix: 预测模型学习得到的权重矩阵
        :return: 每类下的关键基因集合
        """

        # 按权重矩阵中的绝对值从大到小进行排序
        # 选择高排名的前500个基因为该认知得分下的关键基因
        # 对于第一类
        value_sort_1 = abs(np.sort(-abs(weight_matrix[:, 0])))
        value_sort_index_1 = np.argsort(-abs(weight_matrix[:, 0]))

        top_gene_1 = []
        for i in range(500):
            top_gene_1.append(gene[value_sort_index_1[i], :])

        top_gene_1 = np.array(top_gene_1).reshape(500, 2)

        # 对于第二类
        value_sort_2 = abs(np.sort(-abs(weight_matrix[:, 1])))
        value_sort_index_2 = np.argsort(-abs(weight_matrix[:, 1]))

        top_gene_2 = []
        for i in range(500):
            top_gene_2.append(gene[value_sort_index_2[i], :])

        top_gene_2 = np.array(top_gene_2).reshape(500, 2)

        # 对于第三类
        value_sort_3 = abs(np.sort(-abs(weight_matrix[:, 2])))
        value_sort_index_3 = np.argsort(-abs(weight_matrix[:, 2]))

        top_gene_3 = []
        for i in range(500):
            top_gene_3.append(gene[value_sort_index_3[i], :])

        top_gene_3 = np.array(top_gene_3).reshape(500, 2)

        return top_gene_1, top_gene_2, top_gene_3










