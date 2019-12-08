# 导入所有的依赖包
import tensorflow as tf
import numpy as np
import pandas as pd
from CNNmodel import cnnModel
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
import sys

def trasform_data_format(data, label):
    r, c = data.shape

    # 创建空的多维数组用于存放数据
    dataset_array = np.zeros(shape = (c, 148, 148, 1))
    # 创建空的数组用于存放图片的标注信息
    dataset_labels = np.zeros(shape = (c), dtype=np.uint8)
    # 从文件夹下读取数据
    index = 0
    for i in range(c):
        # 格式转化为148x148x1shape
        data_reshaped = np.reshape(data[0:21904, i], newshape=(148, 148, 1))
        # 将维度转换后的图片存入指定的数组内
        dataset_array[index, :, :, :] = data_reshaped
        dataset_labels[index] = label[i]
        index = index + 1

    return dataset_array, dataset_labels


def create_model():
    # 判断是否有预训练模型
    model = cnnModel(0.5)
    model = model.createModel()
    return model

def main():
    # 读入数据
    os.chdir('E:/Experiment/AD_2/preprocess_data/')
    # ========= Step 1. 读入数据 ===========
    isoform_expression_df = pd.read_csv('select_isoform_express.csv')
    isoform_expression = isoform_expression_df.as_matrix()
    isoform_expression = isoform_expression[:, 2:]
    # 对每一列数据进行归一化处理
    scaler = MinMaxScaler()
    isoform_express_scaled = scaler.fit_transform(isoform_expression)

    isoform_label_df = pd.read_csv('clinical_label.csv')
    isoform_label = isoform_label_df.as_matrix()
    isoform_label = isoform_label[:, 1]

    isoform_express_scaled, isoform_label = trasform_data_format(isoform_express_scaled, isoform_label)
    # 对标签数据进行OneHot编码
    isoform_label = tf.keras.utils.to_categorical(isoform_label)
    print(isoform_label.shape)

    isoform_expression_AD_df = pd.read_csv('select_isoform_expression_AD.csv')
    isoform_expression_AD = isoform_expression_AD_df.as_matrix()
    isoform_expression_AD = isoform_expression_AD[:, 2:]
    # 对每一列数据进行归一化处理
    scaler = MinMaxScaler()
    isoform_expression_AD_scaled = scaler.fit_transform(isoform_expression_AD)

    isoform_label_AD_df = pd.read_csv('AD_clinical_label.csv')
    isoform_label_AD = isoform_label_AD_df.as_matrix()
    isoform_label_AD = isoform_label_AD[:, 1]

    isoform_express_scaled_AD, isoform_label_AD = trasform_data_format(isoform_expression_AD_scaled, isoform_label_AD)
    # 对标签数据进行OneHot编码
    isoform_label_AD = tf.keras.utils.to_categorical(isoform_label_AD)

    isoform_expression_ND_df = pd.read_csv('select_isoform_expression_ND.csv')
    isoform_expression_ND = isoform_expression_ND_df.as_matrix()
    isoform_expression_ND = isoform_expression_ND[:, 2:]
    # 对每一列数据进行归一化处理
    scaler = MinMaxScaler()
    isoform_expression_ND_scaled = scaler.fit_transform(isoform_expression_ND)

    isoform_label_ND_df = pd.read_csv('ND_clinical_label.csv')
    isoform_label_ND = isoform_label_ND_df.as_matrix()
    isoform_label_ND = isoform_label_ND[:, 1]

    isoform_express_scaled_ND, isoform_label_ND = trasform_data_format(isoform_expression_ND_scaled, isoform_label_ND)
    # 对标签数据进行OneHot编码
    isoform_label_ND = tf.keras.utils.to_categorical(isoform_label_ND)

    # 调用训练模型
    model1 = create_model( )
    # 用数据训练模型
    model1.fit(isoform_express_scaled, isoform_label, validation_split=0.3, epochs=10, batch_size=1, verbose=1)

    # 调用训练模型
    model2 = create_model( )
    # 用数据训练模型
    model2.fit(isoform_express_scaled_AD, isoform_label_AD, validation_split=0.3, epochs=10, batch_size=1, verbose=1)

    # 调用训练模型
    model3 = create_model( )
    # 用数据训练模型
    model3.fit(isoform_express_scaled_ND, isoform_label_ND, validation_split=0.3, epochs=10, batch_size=1, verbose=1)

    # 使用训练好的模型对所有数据进行预测
    label_prediction = model1.predict(isoform_express_scaled)
    label_prediction_AD = model2.predict(isoform_express_scaled_AD)
    label_prediction_ND = model3.predict(isoform_express_scaled_ND)

    # 使用argmax将预测结果转换为对应的数字索引
    # index = tf.math.argmax(label_predicton[0]).numpy()

    # 保存结果
    label_prediction_df = pd.DataFrame(data=label_prediction)
    label_prediction_AD_df = pd.DataFrame(data=label_prediction_AD)
    label_prediction_ND_df = pd.DataFrame(data=label_prediction_ND)

    label_prediction_df.to_csv('E:/Experiment/AD_2/result/CNN/CNN_label.csv')
    label_prediction_AD_df.to_csv('E:/Experiment/AD_2/result/CNN/CNN_label_AD.csv')
    label_prediction_ND_df.to_csv('E:/Experiment/AD_2/result/CNN/CNN_label_ND.csv')


if __name__ == '__main__':
    main()




