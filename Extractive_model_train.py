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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "00"

num_classes = 2
maxlen = 100
config_path = "roberta/bert_config.json"
checkpoint_path = "roberta/bert_model.ckpt"
dict_path = "roberta/vocab.txt"
train_data_path = "class_data/class_train.json"
test_data_path = "class_data/class_test.json"
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False, )
output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)


# model.summary()
def sparse_categorical_crossentropy(y_true, y_pred):
    """自定义稀疏交叉熵
    这主要是因为keras自带的sparse_categorical_crossentropy不支持求二阶梯度。
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.categorical_crossentropy(y_true, y_pred)


def loss_with_gradient_penalty(y_true, y_pred, epsilon=1):
    """带梯度惩罚的loss
    """
    loss = K.mean(sparse_categorical_crossentropy(y_true, y_pred))
    embeddings = search_layer(y_pred, 'Embedding-Token').embeddings
    gp = K.sum(K.gradients(loss, [embeddings])[0].values ** 2)
    return loss + 0.5 * epsilon * gp


model.compile(
    loss=loss_with_gradient_penalty,
    optimizer=Adam(2e-5),
    metrics=['sparse_categorical_accuracy'],
)


def load_json_data(data_path):
    text_label = []
    with open(data_path, 'r', encoding="utf8") as f:
        for line in tqdm(f):
            data = json.loads(line)
            text_item = data.get('text')
            for item in text_item:
                text, label = item['sentence'], item['label']
                text_label.append((text, int(label)))
    return text_label


train_data = load_json_data(train_data_path)
valid_data = load_json_data(test_data_path)
all_data = train_data + valid_data
print("数据量：训练集、验证集、总数", len(train_data), len(valid_data), len(all_data))
split_total = 20
train_data = numpy.array_split(train_data, split_total)
# valid_data = numpy.array_split(valid_data, split_total)
# print(len(train_data), len(valid_data))


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in tqdm(data):
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        y_true = y_true.astype(int)
        total += len(y_true)
        right += (y_true == y_pred).sum()
        # print(right,total)
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0  ################################################################################################

    def on_epoch_end(self, epoch, logs=None):
        valid_generator = data_generator(valid_data, batch_size)
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model_path = 'class_roberta_best.h5'
            model.save(model_path)  # 保存模型
            print(u'最佳模型，保存成功！！')
        model_path = 'class_robert_most.h5'
        model.save(model_path)  # 保存模型
        print(u'最新模型，保存成功！！')
        print(
            u'目前的准确率: %.5f, 最好的准确率: %.5f\n' %
            (val_acc, self.best_val_acc))


epochs = 1
batch_size = 32  ##############################################################后期修改变小
evaluator = Evaluator()
for i in range(20):
    train_generator = data_generator(train_data[i], batch_size)
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator])
