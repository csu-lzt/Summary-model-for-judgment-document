#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy
import os
import json
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.layers import Lambda, Dense
from keras.models import Model
from bert4keras.backend import keras, search_layer, K
from Evaluate_rouge import evaluate_rouge
from function.segment import split_part

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "00"

root_path = "/root/data/"
config_path = root_path + "roberta/bert_config.json"
checkpoint_path = root_path + "roberta/bert_model.ckpt"
dict_path = root_path + "roberta/vocab.txt"


# 基本参数
maxlen = 512
# 建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'], )
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (title, content) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                content, title, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):  # 交叉熵作为loss，并mask掉输入部分
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)  # 具有整体目标的分类交叉熵
        print(loss.shape)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,
    residual_attention_scores=True,
    hierarchical_position=True,
)
output = CrossEntropy(2)(model.inputs + model.outputs)  # output_axis：2
model = Model(model.inputs, output)
model.compile(optimizer=Adam(4e-5))
print('正在加载模型')
model.load_weights(root_path+'model/segment-summary-20*30bn-5*16bn-5*8bn/model_abstract_best.h5')

class AutoSummary(AutoRegressiveDecoder):
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        # print(u'开始预测一个字符:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        pre = model.predict([token_ids, segment_ids])[:, -1]
        #         print([token_ids, segment_ids])
        return pre

    def generate(self, text, topk):
        textlen = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=textlen)
        # print(u'开始beam search:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk)  # 基于beam search
        # print(u'结束beam search:',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        return tokenizer.decode(output_ids)


def text_abstract(text):
    # 分配文本和摘要的长度
    textlen = min(len(text), 400)
    summarylen = maxlen - textlen
    summarylen = min(textlen, summarylen)
    # print(u'摘要长度：',summarylen)
    topk = 2
    obj = AutoSummary(start_id=None, end_id=tokenizer._token_end_id, maxlen=summarylen)
    summary = obj.generate(text, topk)
    return summary


def generate_whole_summary(label1_list):
    label1_len = 0
    for sentence in label1_list:
        label1_len += len(sentence)
    part_list = split_part(label1_list, min(int(label1_len / 2), 400))
    summary = ''
    for part in part_list:
        part_summary = text_abstract(part)
        summary += part_summary
    return summary


class Evaluate(keras.callbacks.Callback):
    def __init__(self, data):
        self.best_rouge = 0
        self.data = data

    def on_epoch_end(self, epoch, logs=None):
        ROUGE_label1 = np.zeros((4, 3))
        ROUGE_pre_label1 = np.zeros((4, 3))
        i = 0
        for item in tqdm(self.data):
            golden_summary = item[0]
            label1_text = item[1]
            pre_label1_text = item[2]
            summary_label1 = generate_whole_summary(label1_text)
            summary_pre_label1 = generate_whole_summary(pre_label1_text)
            rouge_label1 = evaluate_rouge(summary_label1, golden_summary)
            rouge_pre_label1 = evaluate_rouge(summary_pre_label1, golden_summary)
            print('标准摘要：', golden_summary)
            print('生成摘要：', summary_pre_label1)
            print('rouge-F1',rouge_pre_label1[3][0])
            # !!!!!!这里的rouge已经乘了100，以后习惯
            ROUGE_label1 += rouge_label1
            ROUGE_pre_label1 += rouge_pre_label1
            i += 1
        ROUGE_label1_mean = ROUGE_label1 / i
        ROUGE_pre_label1_mean = ROUGE_pre_label1 / i
        print('根据原始label1生成摘要的ROUGE得分矩阵：', ROUGE_label1_mean)
        print('根据预测label1生成摘要的ROUGE得分矩阵：', ROUGE_pre_label1_mean)
        print('以预测label1摘要ROUGE为准，原始label1作为对比')
        ROUGE_F1 = ROUGE_pre_label1_mean[3][0]
        if ROUGE_F1 > self.best_rouge:
            self.best_rouge = ROUGE_F1
            model_path = root_path + 'model_abstract_best.h5'
            model.save(model_path)  # 保存模型
            print(u'最佳模型，保存成功！！')
        model_path = root_path + 'model_abstract_most.h5'
        model.save(model_path)  # 保存模型
        print(u'最新模型，保存成功！！')
        print(
            u'当前epoch评分: %.5f, 最好评分: %.5f\n' %
            (ROUGE_F1, self.best_rouge))


def load_json_data(data_path, FLAG_test=False):
    data_list = []
    with open(data_path, 'r', encoding="utf8") as f:
        for line in tqdm(f):
            data = json.loads(line)
            summary = data.get('summary')
            text_label1 = data.get('label1')
            if FLAG_test:
                text_pre_label1 = data.get('pre_label1')
                data_list.append((summary, text_label1, text_pre_label1))
            else:
                data_list.append((summary, text_label1))  # 格式：输出的摘要，输入的文本
    return data_list


train_data_path = '/root/data/abstract_data/summary-segment.json'
test_data_path = '/root/data/abstract_data/summary-segment-test.json'
train_data = load_json_data(train_data_path)
test_data = load_json_data(test_data_path, FLAG_test=True)

evaluator = Evaluate(test_data[:50])
batch_size = 8
epochs = 20
train_generator = data_generator(train_data, batch_size)  # data和batchsize  需要看情况修改
model.fit_generator(
    train_generator.forfit(),
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    callbacks=[evaluator]
)
