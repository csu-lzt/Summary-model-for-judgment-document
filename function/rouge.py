#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 针对中分分词列表的rouge函数
import copy
import numpy as np


def Rouge_1(grams_model, grams_reference):  # terms_reference为参考摘要，terms_model为候选摘要   ***one-gram*** 一元模型
    temp_precision = 0  # 准确率（precision） （你给出的结果有多少是正确的）
    temp_recall = 0  # 召回率（recall） （正确的结果有多少被你给出了）
    grams_reference_all = len(grams_reference)
    grams_model_all = len(grams_model)
    for i in range(grams_reference_all):
        c = grams_reference[i - temp_recall]  # remove了多少个出去，就要往前移多少个，确保下标不会出错
        if c in grams_model:
            grams_reference.remove(c)
            grams_model.remove(c)
            temp_recall = temp_recall + 1
    temp_precision = temp_recall
    precision = temp_precision / grams_model_all
    recall = temp_recall / grams_reference_all
    if temp_recall == 0:
        Fscore = 0
    else:
        Fscore = (2 * precision * recall) / (precision + recall)
    # print(u'准确率：',precision)
    # print(u'召回率：',recall)
    # print(u'R1：', Fscore)

    return [Fscore, precision, recall]


def Rouge_2(grams_model, grams_reference):  # terms_reference为参考摘要，terms_model为候选摘要   ***Bi-gram***  2元模型
    temp_precision = 0  # 准确率（precision） （你给出的结果有多少是正确的）
    temp_recall = 0  # 召回率（recall） （正确的结果有多少被你给出了）
    grams_reference_all = len(grams_reference) - 1  # 这里减1代表2元组的个数
    grams_model_all = len(grams_model) - 1
    gram_2_model = []
    gram_2_reference = []
    for x in range(grams_model_all):
        gram_2_model.append(grams_model[x] + grams_model[x + 1])
    for x in range(grams_reference_all):
        gram_2_reference.append(grams_reference[x] + grams_reference[x + 1])

    for i in range(grams_reference_all):
        c = gram_2_reference[i - temp_recall]
        if c in gram_2_model:
            gram_2_reference.remove(c)
            gram_2_model.remove(c)
            temp_recall = temp_recall + 1
    temp_precision = temp_recall
    precision = temp_precision / grams_model_all
    recall = temp_recall / grams_reference_all
    if temp_recall == 0:
        Fscore = 0
    else:
        Fscore = (2 * precision * recall) / (precision + recall)
    #     print(u'准确率：',precision)
    #     print(u'召回率：',recall)
    # print(u'R2：', Fscore)
    return [Fscore, precision, recall]


def LCS(string1, string2):
    len1 = len(string1)
    len2 = len(string2)
    res = [[0 for i in range(len1 + 1)] for j in range(len2 + 1)]
    for i in range(1, len2 + 1):
        for j in range(1, len1 + 1):
            if string2[i - 1] == string1[j - 1]:
                res[i][j] = res[i - 1][j - 1] + 1
            else:
                res[i][j] = max(res[i - 1][j], res[i][j - 1])
    return res[-1][-1]


def Rouge_L(grams_model, grams_reference):  # terms_reference为参考摘要，terms_model为候选摘要   ***one-gram*** 一元模型
    grams_reference_all = len(grams_reference)
    grams_model_all = len(grams_model)
    LCS_n = LCS(grams_model, grams_reference)
    precision = LCS_n / grams_model_all
    recall = LCS_n / grams_reference_all
    if recall == 0:
        Fscore = 0
    else:
        Fscore = (2 * precision * recall) / (precision + recall)
    #     print(u'准确率：',precision)
    #     print(u'召回率：',recall)
    # print(u'RL：', Fscore)
    # print(u'最长子序列', LCS_n)
    return [Fscore, precision, recall]


def Rouge(grams_model, grams_reference):
    grams_model1 = copy.deepcopy(grams_model)
    grams_model2 = copy.deepcopy(grams_model)
    grams_model3 = copy.deepcopy(grams_model)
    grams_reference1 = copy.deepcopy(grams_reference)
    grams_reference2 = copy.deepcopy(grams_reference)
    grams_reference3 = copy.deepcopy(grams_reference)
    rouge_1_F1, rouge_1_precision, rouge_1_recall = Rouge_1(grams_model1, grams_reference1)
    rouge_2_F1, rouge_2_precision, rouge_2_recall = Rouge_2(grams_model2, grams_reference2)
    rouge_L_F1, rouge_L_precision, rouge_L_recall = Rouge_L(grams_model3, grams_reference3)
    rouge_all_F1 = 0.2 * rouge_1_F1 + 0.4 * rouge_2_F1 + 0.4 * rouge_L_F1
    rouge_all_precison = 0.2 * rouge_1_precision + 0.4 * rouge_2_precision + 0.4 * rouge_L_precision
    rouge_all_recall = 0.2 * rouge_1_recall + 0.4 * rouge_2_recall + 0.4 * rouge_L_recall

    rouge_1 = [rouge_1_F1, rouge_1_precision, rouge_1_recall]
    rouge_2 = [rouge_2_F1, rouge_2_precision, rouge_2_recall]
    rouge_L = [rouge_L_F1, rouge_L_precision, rouge_L_recall]
    rouge_all = [rouge_all_F1, rouge_all_precison, rouge_all_recall]
    rouge = np.array([rouge_1, rouge_2, rouge_L, rouge_all])
    return rouge
