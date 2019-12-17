#导入需要依赖的包
import numpy as np
import pandas as pd
import math
import os
from sklearn.preprocessing import MinMaxScaler

# 定义一个tree_top方法类
class tree_top(object):

    # 定义归一化等分函数
    def normalize_partition_t(expression_data_df, t):
        """
        :param expression_data: 表达数据
        :param t: 等分的份数
        :return: 等分修改后的表达数据
        """
        # 对数据进行归一化
        expression_data = expression_data_df.as_matrix()
        col_name = expression_data[0, :]
        row_name = expression_data[1:, :2]
        express_data = expression_data[1:, 2:]
        scaler = MinMaxScaler()
        express_data_scaled = scaler.fit_transform(express_data)

        r, c = express_data_scaled.shape

        partition_express = express_data_scaled

        print("partition isoform expression!")

        # t可取6，12，18，24，30，36，42
        for i in range(r):
            for j in range(c):
                k = 0
                for k in range(t):
                    if express_data_scaled[i, j] >= float(k / t) and express_data_scaled[i, j] <= float((k + 1) / t):
                        partition_express[i, j] = round((k + 1) / t, 6)
                        break

        # 将name和表达数据合并，并返回
        modified_express = np.hstack((row_name, partition_express))
        modified_express = np.vstack((col_name, modified_express))

        return modified_express

    # 定义计算信息增益的函数
    def compute_information_gain(modified_express, label):
        """
        :param modified_express: 归一化划分后的表达数据
        :param label: 标签值
        :return: 所有基因的信息增益列表
        """

        # 对数据进行归一化
        col_name = modified_express[0, 2:]
        row_name = modified_express[1:, :2]
        express_data = modified_express[1:, 2:]

        r, c = express_data.shape

        label = label[:, 1]

        label_all = ['1_1', '1_2', '1_3', '1_4', '1_5', '1_6', '2_1', '2_2', '2_3', '2_4', '2_5', '2_6']
        # 1 代表致病，2代表非致病

        all = 0
        class_1_1 = 0
        class_1_2 = 0
        class_1_3 = 0
        class_1_4 = 0
        class_1_5 = 0
        class_1_6 = 0
        class_2_1 = 0
        class_2_2 = 0
        class_2_3 = 0
        class_2_4 = 0
        class_2_5 = 0
        class_2_6 = 0

        # 计算原始样本集合的信息熵
        print("Compute the information entropy of all samples!")
        Ent = 0
        for i in range(c):
            if label[i] == label_all[0]:
                class_1_1 += 1
                all += 1
            elif label[i] == label_all[1]:
                class_1_2 += 1
                all += 1
            elif label[i] == label_all[2]:
                class_1_3 += 1
                all += 1
            elif label[i] == label_all[3]:
                class_1_4 += 1
                all += 1
            elif label[i] == label_all[4]:
                class_1_5 += 1
                all += 1
            elif label[i] == label_all[5]:
                class_1_6 += 1
                all += 1
            elif label[i] == label_all[6]:
                class_2_1 += 1
                all += 1
            elif label[i] == label_all[7]:
                class_2_2 += 1
                all += 1
            elif label[i] == label_all[8]:
                class_2_3 += 1
                all += 1
            elif label[i] == label_all[9]:
                class_2_4 += 1
                all += 1
            elif label[i] == label_all[10]:
                class_2_5 += 1
                all += 1
            elif label[i] == label_all[11]:
                class_2_6 += 1
                all += 1

        number = [class_1_1, class_1_2, class_1_3, class_1_4, class_1_5, class_1_6, class_2_1, class_2_2, class_2_3,
                  class_2_4, class_2_5, class_2_6]

        for m in range(len(number)):
            if number[m] != 0:
                Ent += - round((number[m] / all) * math.log(float(number[m] / all), 2), 5)

        # 计算每个基因的信息增益
        print("Compute the information gain for each gene!")
        Gain = []
        for i in range(r):
            print("gene ", i)
            Gain_g = 0
            elementary = set(express_data[i, :])
            print(elementary)
            # 取出每个元素对应的样本和标签，用以计算相应分支节点的信息熵
            Ent_mid = []
            weight = []
            for j in elementary:
                print("node", j)
                sample_ = []
                label_ = []
                Ent_ = 0
                Gain_ = 0
                all = 1
                class_1_1 = 0
                class_1_2 = 0
                class_1_3 = 0
                class_1_4 = 0
                class_1_5 = 0
                class_1_6 = 0
                class_2_1 = 0
                class_2_2 = 0
                class_2_3 = 0
                class_2_4 = 0
                class_2_5 = 0
                class_2_6 = 0
                for l in range(c):
                    if express_data[i, l] == j:
                        sample_.append(col_name[l])
                        label_.append(label[l])
                for k in range(len(label_)):
                    if label_[k] == label_all[0]:
                        class_1_1 += 1
                        all += 1
                    elif label_[k] == label_all[1]:
                        class_1_2 += 1
                        all += 1
                    elif label_[k] == label_all[2]:
                        class_1_3 += 1
                        all += 1
                    elif label_[k] == label_all[3]:
                        class_1_4 += 1
                        all += 1
                    elif label_[k] == label_all[4]:
                        class_1_5 += 1
                        all += 1
                    elif label_[k] == label_all[5]:
                        class_1_6 += 1
                        all += 1
                    elif label_[k] == label_all[6]:
                        class_2_1 += 1
                        all += 1
                    elif label_[k] == label_all[7]:
                        class_2_2 += 1
                        all += 1
                    elif label_[k] == label_all[8]:
                        class_2_3 += 1
                        all += 1
                    elif label_[k] == label_all[9]:
                        class_2_4 += 1
                        all += 1
                    elif label_[k] == label_all[10]:
                        class_2_5 += 1
                        all += 1
                    elif label_[k] == label_all[11]:
                        class_2_6 += 1
                        all += 1

                # 计算当前叶子节点的信息熵
                number = [class_1_1, class_1_2, class_1_3, class_1_4, class_1_5, class_1_6, class_2_1, class_2_2,
                          class_2_3,
                          class_2_4, class_2_5, class_2_6]

                for m in range(len(number)):
                    if number[m] != 0:
                        Ent_ += - round((number[m] / all) * math.log(float(number[m] / all), 2), 5)

                Ent_mid.append(Ent_)
                weight.append(all / c)

            # 计算当前叶子节点的信息增益
            for s in range(len(Ent_mid)):
                Gain_ += weight[s] * Ent_mid[s]

            # 计算基因的信息增益
            Gain_g = Ent - Gain_
            Gain.append(Gain_g)

        # 返回每个基因的信息增益
        Gain = np.array(Gain).reshape((r, 1))

        Gene_Gain = np.hstack((row_name, Gain))

        return Gene_Gain

    # 定义计算AD/ND分支的Gini index的函数
    # 这里的modified_express为AD样本的expression data或者ND样本的expression data
    # label 为AD样本的label或者ND样本的label
    def compute_Gini_index(modified_express, label):
        """
        :param modified_express: 归一化分离后的基因表达数据
        :param label: 每个样本的标签
        :return: 每个基因的基尼指数
        """
        # 对数据进行归一化
        col_name = modified_express[0, 2:]
        row_name = modified_express[1:, :2]
        express_data = modified_express[1:, 2:]

        r, c = express_data.shape

        label = label[:, 1]

        label_all = ['1', '2', '3', '4', '5', '6']
        # 1 代表致病，2代表非致病

        all = 0
        class_1_1 = 0
        class_1_2 = 0
        class_1_3 = 0
        class_1_4 = 0
        class_1_5 = 0
        class_1_6 = 0
        # 计算每个基因的基尼指数
        print("Compute the Gini index for each gene!")
        # 有两种划分方式，一种是划分为两类（将认知得分1，2，3合并，将认知得分4，5，6合并）
        Gini_index_ = []
        for i in range(r):
            print("gene ", i)
            Gini_g = 0
            elementary = set(express_data[i, :])
            print(elementary)
            # 取出每个元素对应的样本和标签，用以计算相应分支节点的基尼指数
            Gini_mid = []
            weight = []
            for j in elementary:
                print("node", j)
                sample_ = []
                label_ = []
                Gini_ = 0
                all = 0
                class_1 = 0
                class_2 = 0
                class_3 = 0
                class_4 = 0
                class_5 = 0
                class_6 = 0
                for l in range(c):
                    if express_data[i, l] == j:
                        sample_.append(col_name[l])
                        label_.append(label[l])
                for k in range(len(label_)):
                    if label_[k] == label_all[0]:
                        class_1 += 1
                        all += 1
                    elif label_[k] == label_all[1]:
                        class_2 += 1
                        all += 1
                    elif label_[k] == label_all[2]:
                        class_3 += 1
                        all += 1
                    elif label_[k] == label_all[3]:
                        class_4 += 1
                        all += 1
                    elif label_[k] == label_all[4]:
                        class_5 += 1
                        all += 1
                    elif label_[k] == label_all[5]:
                        class_6 += 1
                        all += 1

                # 计算以当前基因为属性进行叶子节点划分，划分为2类时的参数
                number = [class_1 + class_2 + class_3, class_4 + class_5 + class_6]

                for m in range(len(number)):
                    if number[m] != 0:
                        Gini_ += round((number[m] / all) * (number[m] / all), 5)

                Gini_ = 1 - Gini_

                Gini_mid.append(Gini_)
                weight.append(all / c)

            # 计算以当前基因进行划分为2类的基尼指数
            for s in range(len(Gini_mid)):
                Gini_g += weight[s] * Gini_mid[s]

            Gini_index_.append(Gini_g)

        Gini_index = np.array(Gini_index_).reshape(r, 1)
        Gene_Gini = np.hstack((row_name, Gini_index))

        return Gene_Gini

    # 定义计算左分支的Gini index的函数
    # 这里的modified_express为AD/ND样本的认知得分为1，2，3的expression data
    # label 为AD/ND样本中认知得分为1，2，3的label
    def compute_Gini_index_l(modified_express, label):
        """
        :param modified_express: 归一化分离后的基因表达数据
        :param label: 每个样本的标签
        :return: 每个基因的基尼指数
        """
        # 对数据进行归一化
        col_name = modified_express[0, 2:]
        row_name = modified_express[1:, :2]
        express_data = modified_express[1:, 2:]

        r, c = express_data.shape

        label = label[:, 1]

        label_all = ['1', '2', '3']

        all = 0
        class_1 = 0
        class_2 = 0
        class_3 = 0

        # 计算每个基因的基尼指数
        print("Compute the Gini index for each gene!")
        Gini_index = []
        for i in range(r):
            print("gene ", i)
            Gini_g = 0
            elementary = set(express_data[i, :])
            print(elementary)
            # 取出每个元素对应的样本和标签，用以计算相应分支节点的基尼指数
            Gini_mid = []
            weight = []
            for j in elementary:
                print("node", j)
                sample_ = []
                label_ = []
                Gini_ = 0
                all = 0
                class_1 = 0
                class_2 = 0
                class_3 = 0
                for l in range(c):
                    if express_data[i, l] == j:
                        sample_.append(col_name[l])
                        label_.append(label[l])
                for k in range(len(label_)):
                    if label_[k] == label_all[0]:
                        class_1 += 1
                        all += 1
                    elif label_[k] == label_all[1]:
                        class_2 += 1
                        all += 1
                    elif label_[k] == label_all[2]:
                        class_3 += 1
                        all += 1

                # 计算以当前基因为属性进行叶子节点划分，划分为2类时的参数
                number = [class_1, class_2, class_3]

                for m in range(len(number)):
                    if number[m] != 0:
                        Gini_ += round((number[m] / all) * (number[m] / all), 5)

                Gini_ = 1 - Gini_

                Gini_mid.append(Gini_)
                weight.append(all / c)

            # 计算以当前基因进行划分的基尼指数
            for s in range(len(Gini_mid)):
                Gini_g += weight[s] * Gini_mid[s]

            Gini_index.append(Gini_g)

        Gini_index = np.array(Gini_index).reshape(r, 1)
        Gene_Gini = np.hstack((row_name, Gini_index))

        return Gene_Gini

    # 定义计算右分支的Gini index的函数
    # 这里的modified_express为AD/ND样本的认知得分为4，5，6的expression data
    # label 为AD/ND样本中认知得分为4，5，6的label
    def compute_Gini_index_r(modified_express, label):
        """
        :param modified_express: 归一化分离后的基因表达数据
        :param label: 每个样本的标签
        :return: 每个基因的基尼指数
        """
        # 对数据进行归一化
        col_name = modified_express[0, 2:]
        row_name = modified_express[1:, :2]
        express_data = modified_express[1:, 2:]

        r, c = express_data.shape

        label = label[:, 1]

        label_all = ['4', '5', '6']

        all = 0
        class_4 = 0
        class_5 = 0
        class_6 = 0

        # 计算每个基因的基尼指数
        print("Compute the Gini index for each gene!")
        Gini_index = []
        for i in range(r):
            print("gene ", i)
            Gini_g = 0
            elementary = set(express_data[i, :])
            print(elementary)
            # 取出每个元素对应的样本和标签，用以计算相应分支节点的基尼指数
            Gini_mid = []
            weight = []
            for j in elementary:
                print("node", j)
                sample_ = []
                label_ = []
                Gini_ = 0
                all = 0
                class_4 = 0
                class_5 = 0
                class_6 = 0
                for l in range(c):
                    if express_data[i, l] == j:
                        sample_.append(col_name[l])
                        label_.append(label[l])
                for k in range(len(label_)):
                    if label_[k] == label_all[0]:
                        class_4 += 1
                        all += 1
                    elif label_[k] == label_all[1]:
                        class_5 += 1
                        all += 1
                    elif label_[k] == label_all[2]:
                        class_6 += 1
                        all += 1

                # 计算以当前基因为属性进行叶子节点划分，划分为2类时的参数
                number = [class_4, class_5, class_6]

                for m in range(len(number)):
                    if number[m] != 0:
                        Gini_ += round((number[m] / all) * (number[m] / all), 5)

                Gini_ = 1 - Gini_

                Gini_mid.append(Gini_)
                weight.append(all / c)

            # 计算以当前基因进行划分的基尼指数
            for s in range(len(Gini_mid)):
                Gini_g += weight[s] * Gini_mid[s]

            Gini_index.append(Gini_g)

        Gini_index = np.array(Gini_index).reshape(r, 1)
        Gene_Gini = np.hstack((row_name, Gini_index))

        return Gene_Gini

    # 按指标从大到小进行排序，根据排序表选择高排名的基因
    def rank_top_down(gene_value):
        """
        :param gene_value: gene及其对应的度量值
        :return: 高排名的基因名字列表
        """
        r, c = gene_value.shape

        # 按从大到小进行排序，并记录排名对应的索引
        value_sort = abs(np.sort( -gene_value[:, 2]))
        value_sort_index = np.argsort(-gene_value[:, 2])

        top_gene_sort = []
        for i in range(r):
            top_gene_sort.append(gene_value[value_sort_index[i], :])

        top_gene_sort = np.array(top_gene_sort).reshape(r, 3)

        return top_gene_sort

    # 按指标从小到大进行排序，根据排序表选择高排名的基因
    def rank_down_top(gene_value):
        """
        :param gene_value: gene及其对应的度量值
        :return: 高排名的基因名字列表
        """
        r, c = gene_value.shape

        # 按从小到大进行排序，并记录排名对应的索引
        value_sort = abs(np.sort(gene_value[:, 2]))
        value_sort_index = np.argsort(gene_value[:, 2])

        top_gene_sort = []
        for i in range(r):
            top_gene_sort.append(gene_value[value_sort_index[i], :])

        top_gene_sort = np.array(top_gene_sort).reshape(r, 3)

        return top_gene_sort

    # 根据基因列表筛选基因
    def select(reference_gene, gene_expression):
        """
        :param reference_gene: 需要提取的基因
        :param gene_expression_df:  基因表达数据
        :return: 提取的基因表达数据
        """
        select_gene = []

        r_r, c_r = reference_gene.shape

        r_g, c_g = gene_expression.shape

        for i in range(r_r):
            for j in range(r_g):
                if gene_expression[j, 0] == reference_gene[i, 0]:
                    select_gene.append(gene_expression[j, :])
                    break

        select_gene = np.array(select_gene).reshape(r_r, c_g)

        return select_gene