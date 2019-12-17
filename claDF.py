# -*- coding: utf-8 -*-
"""
作者：jx
日期:2019-12-10
版本：1
文件名：claDF.py
功能：构建一棵树进行标签预测和关键基因选择
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import math
import os
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from Tree_down import tree_down
from Tree_top import tree_top



def main():
    # 读入数据
    os.chdir('E:/Experiment/AD_2/preprocess_data/')
    # ========== Step 1. layer 1, 确定用于一棵树输入的节点数 ============
    # 定义划分等分的个数t, t = 600,800,1000,2000,3000,4000,5000,6000,7000,8000,9000
    t = 8000
    # 所有样本的isoform表达数据
    print("Process all isoforms!")
    isoform_expression_df = pd.read_csv('select_isoform_express.csv')
    # 归一化等分后的表达数据
    isoform_expression = tree_top.normalize_partition_t(isoform_expression_df, t)

    # 读入相应的标签值
    clinical_label_df = pd.read_csv('clinical_label.csv')
    clinical_label = clinical_label_df.as_matrix()

    # 计算所有基因的信息增益
    infor_gain_all_gene = tree_top.compute_information_gain(isoform_expression, clinical_label)

    # 从高排名的10000个基因中随机选择高排名的8000个基因作为顶层节点的输入
    rank_list = tree_top.rank_top_down(infor_gain_all_gene)
    top_10000 = rank_list[0:10000, 0]
    top_8000 = np.random.choice(top_10000, 8000)
    top_8000 = np.array(top_8000).reshape(8000, 1)

    # =============== Step 2. layer 2, 确定第二层节点左右分支节点的基因数 ============
    # 确定左分支节点的基因数
    # AD样本的isoform表达数据
    print("Process AD isoforms!")
    isoform_expression_AD_df = pd.read_csv('select_isoform_expression_AD.csv')
    # 归一化等分后的表达数据
    isoform_expression_AD = tree_top.normalize_partition_t(isoform_expression_AD_df, t)
    isoform_expression_AD_select = tree_top.select(top_8000, isoform_expression_AD)

    # 读入相应的标签值
    AD_clinical_label_df = pd.read_csv('AD_clinical_label.csv')
    AD_clinical_label = AD_clinical_label_df.as_matrix()

    # 确定左分支节点的基因数
    ad_Gini_index = tree_top.compute_Gini_index(isoform_expression_AD_select, AD_clinical_label)
    ad_node_rank_list = tree_top.rank_down_top(ad_Gini_index)
    ad_top_6000 = ad_node_rank_list[0:6000, 0]
    ad_top_5000 = np.random.choice(ad_top_6000, 5000)
    ad_top_5000 = np.array(ad_top_5000).reshape(5000, 1)

    # ND样本的isoform表达数据
    print("Process ND isoforms!")
    isoform_expression_ND_df = pd.read_csv('select_isoform_expression_ND.csv')
    # 归一化等分后的表达数据
    isoform_expression_ND = tree_top.normalize_partition_t(isoform_expression_ND_df, t)
    isoform_expression_ND_select = tree_top.select(ad_top_5000, isoform_expression_ND)

    # 读入相应的标签值
    ND_clinical_label_df = pd.read_csv('ND_clinical_label.csv')
    ND_clinical_label = ND_clinical_label_df.as_matrix()

    # 确定右分支节点的基因数
    nd_Gini_index= tree_top.compute_Gini_index(isoform_expression_ND_select, ND_clinical_label)
    nd_node_rank_list = tree_top.rank_down_top(nd_Gini_index)
    nd_top_6000 = nd_node_rank_list[0:6000, 0]
    nd_top_5000 = np.random.choice(nd_top_6000, 5000)
    nd_top_5000 = np.array(nd_top_5000).reshape(5000, 1)

    # =============== Step 3. layer 3, 确定第三层节点中AD/ND分支的左右分支节点的基因数 ============
    # 确定AD分支中左右分支节点的基因数
    # AD123样本的isoform表达数据
    print("Process AD left isoforms!")
    isoform_expression_ADl_df = pd.read_csv('select_isoform_expression_AD123.csv')
    # 归一化等分后的表达数据
    isoform_expression_ADl = tree_top.normalize_partition_t(isoform_expression_ADl_df, t)
    isoform_expression_ADl_select = tree_top.select(ad_top_5000, isoform_expression_ADl)

    # 读入相应的标签值
    ADl_clinical_label_df = pd.read_csv('AD_clinical_label123.csv')
    ADl_clinical_label = ADl_clinical_label_df.as_matrix()

    # 确定左分支节点的基因数
    adl_Gini_index = tree_top.compute_Gini_index(isoform_expression_ADl_select, ADl_clinical_label)
    adl_node_rank_list = tree_top.rank_down_top(adl_Gini_index)
    adl_top_3000 = adl_node_rank_list[0:3000, 0]
    adl_top_1500 = np.random.choice(adl_top_3000, 1500)
    adl_top_1500 = np.array(adl_top_1500).reshape(1500, 1)

    # AD456样本的isoform表达数据
    print("Process AD right isoforms!")
    isoform_expression_ADr_df = pd.read_csv('select_isoform_expression_AD456.csv')
    # 归一化等分后的表达数据
    isoform_expression_ADr = tree_top.normalize_partition_t(isoform_expression_ADr_df, t)
    isoform_expression_ADr_select = tree_top.select(ad_top_5000, isoform_expression_ADr)

    # 读入相应的标签值
    ADr_clinical_label_df = pd.read_csv('AD_clinical_label456.csv')
    ADr_clinical_label = ADr_clinical_label_df.as_matrix()

    # 确定右分支节点的基因数
    adr_Gini_index = tree_top.compute_Gini_index(isoform_expression_ADr_select, ADr_clinical_label)
    adr_node_rank_list = tree_top.rank_down_top(adr_Gini_index)
    adr_top_3000 = adr_node_rank_list[0:3000, 0]
    adr_top_1500 = np.random.choice(adr_top_3000, 1500)
    adr_top_1500 = np.array(adr_top_1500).reshape(1500, 1)

    # 确定ND分支中左右分支节点的基因数
    # ND123样本的isoform表达数据
    print("Process ND left isoforms!")
    isoform_expression_NDl_df = pd.read_csv('select_isoform_expression_ND123.csv')
    # 归一化等分后的表达数据
    isoform_expression_NDl = tree_top.normalize_partition_t(isoform_expression_NDl_df, t)
    isoform_expression_NDl_select = tree_top.select(nd_top_5000, isoform_expression_NDl)

    # 读入相应的标签值
    NDl_clinical_label_df = pd.read_csv('ND_clinical_label123.csv')
    NDl_clinical_label = NDl_clinical_label_df.as_matrix()

    # 确定左分支节点的基因数
    ndl_Gini_index = tree_top.compute_Gini_index(isoform_expression_NDl_select, NDl_clinical_label)
    ndl_node_rank_list = tree_top.rank_down_top(ndl_Gini_index)
    ndl_top_3000 = ndl_node_rank_list[0:3000, 0]
    ndl_top_1500 = np.random.choice(ndl_top_3000, 1500)
    ndl_top_1500 = np.array(ndl_top_1500).reshape(1500, 1)

    # ND456样本的isoform表达数据
    print("Process ND right isoforms!")
    isoform_expression_NDr_df = pd.read_csv('select_isoform_expression_ND456.csv')
    # 归一化等分后的表达数据
    isoform_expression_NDr = tree_top.normalize_partition_t(isoform_expression_NDr_df, t)
    isoform_expression_NDr_select = tree_top.select(nd_top_5000, isoform_expression_NDr)

    # 读入相应的标签值
    NDr_clinical_label_df = pd.read_csv('ND_clinical_label456.csv')
    NDr_clinical_label = NDr_clinical_label_df.as_matrix()

    # 确定右分支节点的基因数
    ndr_Gini_index = tree_top.compute_Gini_index(isoform_expression_NDr_select, NDr_clinical_label)
    ndr_node_rank_list = tree_top.rank_down_top(ndr_Gini_index)
    ndr_top_3000 = ndr_node_rank_list[0:3000, 0]
    ndr_top_1500 = np.random.choice(ndr_top_3000, 1500)
    ndr_top_1500 = np.array(ndr_top_1500).reshape(1500, 1)

    # =============== Step 4. layer 4, 确定第四层节点中AD/ND下的每个分支节点对应的关键基因集合 ============
    # 得到每个分支的表达数据及样本标签
    ad1_isoform_express = tree_top.select(adl_top_1500, isoform_expression_ADl)
    adl_name = ad1_isoform_express[:, :2]
    ad1_isoform_express = ad1_isoform_express[:, 2:]
    adl_label = ADl_clinical_label[:, 1]
    adl_isoform_express, adl_sample_label = tree_down.trasform_data_format(ad1_isoform_express, adl_label)
    # 对标签数据进行OneHot编码
    ad1_sample_label_ = tf.keras.utils.to_categorical(adl_sample_label)
    print(ad1_sample_label_)
    ad1_sample_label_ = ad1_sample_label_[:, 1:]

    adr_isoform_express = tree_top.select(adl_top_1500, isoform_expression_ADr)
    adr_name = adr_isoform_express[:, :2]
    adr_isoform_express = adr_isoform_express[:, 2:]
    adr_label = ADr_clinical_label[:, 1]
    adr_isoform_express, adr_sample_label = tree_down.trasform_data_format(adr_isoform_express, adr_label)
    # 对标签数据进行OneHot编码
    adr_sample_label_ = tf.keras.utils.to_categorical(adr_sample_label)
    adr_sample_label_ = adr_sample_label_[:, 4:]

    nd1_isoform_express = tree_top.select(ndl_top_1500, isoform_expression_NDl)
    ndl_name = nd1_isoform_express[:, :2]
    nd1_isoform_express = nd1_isoform_express[:, 2:]
    nd1_label = NDl_clinical_label[:, 1]
    ndl_isoform_express, ndl_sample_label = tree_down.trasform_data_format(nd1_isoform_express, nd1_label)
    # 对标签数据进行OneHot编码
    nd1_sample_label_ = tf.keras.utils.to_categorical(ndl_sample_label)
    nd1_sample_label_ = nd1_sample_label_[:, 1:]

    ndr_isoform_express = tree_top.select(ndr_top_1500, isoform_expression_NDr)
    ndr_name = ndr_isoform_express[:, :2]
    ndr_isoform_express = ndr_isoform_express[:, 2:]
    ndr_label = NDr_clinical_label[:, 1]
    ndr_isoform_express, ndr_sample_label = tree_down.trasform_data_format(ndr_isoform_express, ndr_label)
    # 对标签数据进行OneHot编码
    ndr_sample_label_ = tf.keras.utils.to_categorical(ndr_sample_label)
    ndr_sample_label_ = ndr_sample_label_[:, 4:]

    # 用每个叶子节点上的数据训练模型
    # 对于ad left node
    model_adl = tree_down.Perceptron()
    # 用数据训练模型
    model_adl.fit(adl_isoform_express, ad1_sample_label_, validation_split=0.3, epochs=30, batch_size=1, verbose=1)

    # 用模型进行预测
    adl_predict = model_adl.predict(adl_isoform_express)

    # 保存权重
    adl_weights = model_adl.get_weights()
    weights = adl_weights[0]
    key_gene_adl_1, key_gene_adl_2, key_gene_adl_3 = tree_down.key_gene_set(adl_name, weights)

    # 对于ad right node
    model_adr = tree_down.Perceptron()
    # 用数据训练模型
    model_adr.fit(adr_isoform_express, adr_sample_label_, validation_split=0.3, epochs=30, batch_size=1, verbose=1)

    # 用模型进行预测
    adr_predict = model_adr.predict(adr_isoform_express)

    # 保存权重
    adr_weights = model_adr.get_weights()
    weights = adr_weights[0]
    key_gene_adr_4, key_gene_adr_5, key_gene_adr_6 = tree_down.key_gene_set(adr_name, weights)

    # 对于nd left node
    model_ndl = tree_down.Perceptron()
    # 用数据训练模型
    model_ndl.fit(ndl_isoform_express, nd1_sample_label_, validation_split=0.3, epochs=30, batch_size=1, verbose=1)

    # 用模型进行预测
    ndl_predict = model_ndl.predict(ndl_isoform_express)

    # 保存权重
    ndl_weights = model_ndl.get_weights()
    weights = ndl_weights[0]
    key_gene_ndl_1, key_gene_ndl_2, key_gene_ndl_3 = tree_down.key_gene_set(ndl_name, weights)

    # 对于nd right node
    model_ndr = tree_down.Perceptron()
    # 用数据训练模型
    model_ndr.fit(ndr_isoform_express, ndr_sample_label_, validation_split=0.3, epochs=30, batch_size=1, verbose=1)

    # 用模型进行预测
    ndr_predict = model_ndr.predict(ndr_isoform_express)

    # 保存权重
    ndr_weights = model_ndr.get_weights()
    weights = ndr_weights[0]
    key_gene_ndr_4, key_gene_ndr_5, key_gene_ndr_6 = tree_down.key_gene_set(ndr_name, weights)

    # ==================== Step 5. 保存结果  =====================
    # 保存结果
    adl_predict_df = pd.DataFrame(data = adl_predict)
    adr_predict_df = pd.DataFrame(data = adr_predict)
    ndl_predict_df = pd.DataFrame(data = ndl_predict)
    ndr_predict_df = pd.DataFrame(data = ndr_predict)

    key_gene_adl_1_df = pd.DataFrame(data = key_gene_adl_1)
    key_gene_adl_2_df = pd.DataFrame(data = key_gene_adl_2)
    key_gene_adl_3_df = pd.DataFrame(data = key_gene_adl_3)
    key_gene_adr_4_df = pd.DataFrame(data = key_gene_adr_4)
    key_gene_adr_5_df = pd.DataFrame(data = key_gene_adr_5)
    key_gene_adr_6_df = pd.DataFrame(data = key_gene_adr_6)
    key_gene_ndl_1_df = pd.DataFrame(data = key_gene_ndl_1)
    key_gene_ndl_2_df = pd.DataFrame(data = key_gene_ndl_2)
    key_gene_ndl_3_df = pd.DataFrame(data = key_gene_ndl_3)
    key_gene_ndr_4_df = pd.DataFrame(data = key_gene_ndr_4)
    key_gene_ndr_5_df = pd.DataFrame(data = key_gene_ndr_5)
    key_gene_ndr_6_df = pd.DataFrame(data = key_gene_ndr_6)

    adl_predict_df.to_csv('E:/Experiment/AD_2/result/claDF/t8000/tree6/adl_predict.csv')
    adr_predict_df.to_csv('E:/Experiment/AD_2/result/claDF/t8000/tree6/adr_predict.csv')
    ndl_predict_df.to_csv('E:/Experiment/AD_2/result/claDF/t8000/tree6/ndl_predict.csv')
    ndr_predict_df.to_csv('E:/Experiment/AD_2/result/claDF/t8000/tree6/ndr_predict.csv')

    key_gene_adl_1_df.to_csv('E:/Experiment/AD_2/result/claDF/t8000/tree6/key_gene_adl_1.csv')
    key_gene_adl_2_df.to_csv('E:/Experiment/AD_2/result/claDF/t8000/tree6/key_gene_adl_2.csv')
    key_gene_adl_3_df.to_csv('E:/Experiment/AD_2/result/claDF/t8000/tree6/key_gene_adl_3.csv')
    key_gene_adr_4_df.to_csv('E:/Experiment/AD_2/result/claDF/t8000/tree6/key_gene_adr_4.csv')
    key_gene_adr_5_df.to_csv('E:/Experiment/AD_2/result/claDF/t8000/tree6/key_gene_adr_5.csv')
    key_gene_adr_6_df.to_csv('E:/Experiment/AD_2/result/claDF/t8000/tree6/key_gene_adr_6.csv')
    key_gene_ndl_1_df.to_csv('E:/Experiment/AD_2/result/claDF/t8000/tree6/key_gene_ndl_1.csv')
    key_gene_ndl_2_df.to_csv('E:/Experiment/AD_2/result/claDF/t8000/tree6/key_gene_ndl_2.csv')
    key_gene_ndl_3_df.to_csv('E:/Experiment/AD_2/result/claDF/t8000/tree6/key_gene_ndl_3.csv')
    key_gene_ndr_4_df.to_csv('E:/Experiment/AD_2/result/claDF/t8000/tree6/key_gene_ndr_4.csv')
    key_gene_ndr_5_df.to_csv('E:/Experiment/AD_2/result/claDF/t8000/tree6/key_gene_ndr_5.csv')
    key_gene_ndr_6_df.to_csv('E:/Experiment/AD_2/result/claDF/t8000/tree6/key_gene_ndr_6.csv')

if __name__ == '__main__':
    main()