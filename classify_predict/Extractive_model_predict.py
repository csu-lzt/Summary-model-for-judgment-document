import os
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
from bert4keras.snippets import to_array

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "00"
num_classes = 2
maxlen = 100
rootPath = "/root/data/"
config_path = rootPath + "roberta/bert_config.json"
checkpoint_path = rootPath + "roberta/bert_model.ckpt"
dict_path = rootPath + "roberta/vocab.txt"
model_path = rootPath + "model/class_roberta_best.h5"

tokenizer = Tokenizer(dict_path, do_lower_case=True)
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
model.load_weights(model_path)



def pred_input(sentence):
    maxlen = 100
    token_ids, segment_ids = tokenizer.encode(sentence)
    token_ids, segment_ids = to_array([token_ids[:maxlen]], [segment_ids[:maxlen]])
    rate_label_0, rate_label_1 = model.predict([token_ids, segment_ids])[0]
    if rate_label_0 > rate_label_1:
        label = 0
    else:
        label = 1
    return label


if __name__ == "__main__":
    label = pred_input('综上，根据《中华人民共和abababab阿巴巴规定，判决如下：')
    print(label)
