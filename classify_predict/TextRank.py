#!/usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
import numpy
import jieba
import codecs
import os
import re


class SummaryTxt:  # 摘要文本类
    def __init__(self, stopwordspath):
        # 单词数量
        self.N = 100
        # 单词间的距离（阈值为4或5）
        self.CLUSTER_THRESHOLD = 5
        # 返回的top n句子
        self.TOP_SENTENCES = 8
        self.stopwords = {}
        # 加载停用词
        if os.path.exists(stopwordspath):
            stoplist = [line.strip() for line in codecs.open(stopwordspath, 'r', encoding='utf8').readlines()]
            self.stopwords = {}.fromkeys(stoplist)  # fromkeys() 函数用于创建一个新字典
        else:
            print("no stopwords!!")

    def _score_sentences(self, sentences, topn_words):
        # 利用前N个关键字给句子打分
        # :param sentences: 句子列表
        # :param topn_words: 关键字列表
        # :return:
        score100 = 0
        scores = []
        sentence_idx = -1
        for s in [list(jieba.cut(s)) for s in sentences]:
            sentence_idx += 1
            word_idx = []
            for w in topn_words:
                try:
                    word_idx.append(s.index(w))  # 关键词出现在该句子中的索引位置
                except ValueError:  # w不在句子中
                    pass
            word_idx.sort()
            if len(word_idx) == 0:
                max_cluster_score = 0
            else:
                # 对于两个连续的单词，利用单词位置索引，通过距离阀值计算族
                clusters = []
                cluster = [word_idx[0]]
                i = 1
                while i < len(word_idx):
                    if word_idx[i] - word_idx[i - 1] < self.CLUSTER_THRESHOLD:
                        cluster.append(word_idx[i])
                    else:
                        clusters.append(cluster[:])
                        cluster = [word_idx[i]]
                    i += 1
                clusters.append(cluster)
                # 对每个族打分，每个族类的最大分数是对句子的打分
                max_cluster_score = 0
                for c in clusters:
                    significant_words_in_cluster = len(c)
                    total_words_in_cluster = c[-1] - c[0] + 1
                    score = 1.0 * significant_words_in_cluster * significant_words_in_cluster / total_words_in_cluster
                    if score > max_cluster_score:
                        max_cluster_score = score

            # 自己加的，强制高低分
            sent_text = sentences[sentence_idx]

            if re.search(r"诉讼请求：", sent_text) or re.search(r"本院认为", sent_text) or re.search(r"本院难以支持", sent_text) \
                    or re.search(r"诉讼请求为：", sent_text) or re.search(r"判令：", sent_text) \
                    or re.search(r"诉请：", sent_text) or re.search(r"诉讼为：", sent_text) \
                    or re.search(r"诉讼主张为：", sent_text) or re.search(r"诉请判令", sent_text) \
                    or (re.search(r"法院", sent_text) and re.search(r"诉", sent_text) and re.search(r"求", sent_text)) \
                    or (re.search(r"法院", sent_text) and re.search(r"请", sent_text) and re.search(r"判",
                                                                                                 sent_text)):  # 要求判令/请求判令 合二为一为求判令：
                if len(sent_text) > 10:
                    max_cluster_score = 100
            if re.search(r"诉讼请求：", sentences[sentence_idx - 1]) or re.search(r"诉请：", sentences[sentence_idx - 1]) \
                    or re.search(r"诉讼请求为：", sentences[sentence_idx - 1]) or re.search(r"判令：", sentences[
                sentence_idx - 1]):  # 要求判令/请求判令 合二为一为求判令：
                if sent_text.startswith(("1", "2")):
                    max_cluster_score = 100
            if re.search(r"诉讼请求：", sentences[sentence_idx - 2]) or re.search(r"诉请：", sentences[sentence_idx - 2]) \
                    or re.search(r"诉讼请求为：", sentences[sentence_idx - 2]) or re.search(r"判令：", sentences[
                sentence_idx - 2]):  # 要求判令/请求判令 合二为一为求判令：
                if sent_text.startswith(("2", "3")):
                    max_cluster_score = 100
            if re.search(r"诉讼请求：", sentences[sentence_idx - 3]) or re.search(r"诉请：", sentences[sentence_idx - 3]) \
                    or re.search(r"诉讼请求为：", sentences[sentence_idx - 3]) or re.search(r"判令：", sentences[
                sentence_idx - 3]):  # 要求判令/请求判令 合二为一为求判令：
                if sent_text.startswith(("3", "4")):
                    max_cluster_score = 100

            # 首句改写
            if re.search(r"一审民事判决书", sent_text) and sentence_idx < 2:
                if re.search(r"继承纠纷", sent_text):
                    sentences[sentence_idx] = r"原被告系继承纠纷。"
                    max_cluster_score = 100
                elif re.search(r"借款合同", sent_text):
                    sentences[sentence_idx] = r"原被告系借款合同关系。"
                    max_cluster_score = 100
                elif re.search(r"租赁合同", sent_text):
                    sentences[sentence_idx] = r"原被告系租赁合同关系。"
                    max_cluster_score = 100
                elif re.search(r"侵权责任", sent_text):
                    sentences[sentence_idx] = r"原被告系侵权责任纠纷。"
                    max_cluster_score = 100
                elif re.search(r"劳动合同", sent_text):
                    sentences[sentence_idx] = r"原被告系劳动合同关系。"
                    max_cluster_score = 100
                else:
                    max_cluster_score = 0

            # 判决如下及后面的一、二、三、等
            if re.search(r"判决如下", sent_text):
                max_cluster_score = 100
            if re.search(r"判决如下", sentences[sentence_idx - 1]) and sent_text.startswith((r"一", r"二", r"如")):
                max_cluster_score = 100
            if re.search(r"判决如下", sentences[sentence_idx - 2]) and sent_text.startswith((r"二", r"三", r"如")):
                max_cluster_score = 100
            if re.search(r"判决如下", sentences[sentence_idx - 3]) and sent_text.startswith((r"三", r"四", r"如")):
                max_cluster_score = 100
            if re.search(r"判决如下", sentences[sentence_idx - 4]) and sent_text.startswith((r"四", r"五", r"如")):
                max_cluster_score = 100

            if sentences[sentence_idx - 1].endswith((r"判决如下：", r"判决如下:", r"判决如下;", r"判决如下；")):
                max_cluster_score = 100

            if re.search(r"事实和理由：", sent_text) or re.search(r"经审理", sent_text) or re.search(r"经查明", sent_text):
                max_cluster_score = 50
            # if re.search(r"民事判决书", sent_text) or re.search(r"民 事 判 决 书", sent_text) \
            #         or re.search(r"（2018）", sent_text) or re.search(r"（2017）", sent_text):
            #     max_cluster_score = 0
            scores.append((sentence_idx, max_cluster_score))
            if max_cluster_score == 100:
                score100 += 1
        return scores, score100

    def summaryTopNtxt(self, text):
        # 将文章分成句子
        sentences = text
        # 根据句子列表生成分词列表
        words = [w for sentence in sentences for w in jieba.cut(sentence) if w not in self.stopwords if
                 len(w) > 1 and w != '\t']
        # 统计词频
        wordfre = nltk.FreqDist(words)
        # 获取词频最高的前N个词
        topn_words = [w[0] for w in sorted(wordfre.items(), key=lambda d: d[1], reverse=True)][:self.N]
        # 根据最高的n个关键词，给句子打分
        scored_sentences, score100 = self._score_sentences(sentences, topn_words)
        if score100 > self.TOP_SENTENCES:
            self.TOP_SENTENCES = score100
        top_n_scored = sorted(scored_sentences, key=lambda s: s[1])[-self.TOP_SENTENCES:]
        top_n_scored = sorted(top_n_scored, key=lambda s: s[0])
        summarySentences = ''
        for (idx, score) in top_n_scored:
            # print(sentences[idx])
            summarySentences += sentences[idx]
        while len(summarySentences) < 320:
            self.TOP_SENTENCES += 1
            top_n_scored = sorted(scored_sentences, key=lambda s: s[1])[-self.TOP_SENTENCES:]
            top_n_scored = sorted(top_n_scored, key=lambda s: s[0])
            summarySentences = ''
            for (idx, score) in top_n_scored:
                # print(sentences[idx])
                summarySentences += sentences[idx]
        return summarySentences


def textRank_extract(text):  # "text": [{"sentence":"001"},{"sentence":"002"}]
    text_sentence = []
    for i, _ in enumerate(text):
        sent_text = text[i]["sentence"]
        text_sentence.append(sent_text)  # 原始文本字符串
    obj = SummaryTxt('stopwords.txt')
    summary = obj.summaryTopNtxt(text_sentence)  # 抽取式摘要选n个句子
    return summary
