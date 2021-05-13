#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import jieba

# 这里有问题，是CAIL阶段遗留下的且未被发现的，注意不要犯错误（把逗号等符号去掉了再分词）

# 使用jieba进行分词
def str2list(str):
    # punctuation = """。：，、《》（）"""
    # re_punctuation = "[{}]+".format(punctuation)
    # str = str.replace(" ", "")
    # str = re.sub(re_punctuation, "", str)

    # 正则式去除括号内的内容
    re_str = re.sub(r'[\(|（|【].*?[\)|）|】]', "", str)
    # 一个标点符号表，分词后去除句子中的标点符号
    remove_list = [' ', ',', '.', '，', '。', ':', '：', '《', '》', '(', ')', '（', '）', '、']
    grams_remove = []
    grams = jieba.lcut(re_str)  # 直接切分成列表
    for gram in grams:
        if gram not in remove_list:
            grams_remove.append(gram)
    return grams_remove

# temp=[]
# grams = str2list("我是的计    算    客服号，但是《刷卡缴费》不断得到美。丽的收款方法的服务")
# grams2 = str2list("我是小朋友，但是不断得到美丽的收款方法的服务")
# temp.append(grams)
# temp.append(grams2)
# gramss = grams+grams2
# print(gramss)