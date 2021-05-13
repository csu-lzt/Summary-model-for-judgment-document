# import sys
# import os
#
# curPath = os.path.abspath(os.path.dirname(__file__))
# rootPath = os.path.split(curPath)[0]
# sys.path.append(rootPath)
# # ---------------------------------------------------
# ************************注意修改这个文件**************
# from keras_textclassification.m01_FastText.predict import pred_input
# from Extractive_model_predict import pred_input
import json
import numpy as np
from Evaluate_rouge import evaluate_rouge
from tqdm import tqdm

# print(pred_input('综上，根据《中华人民共和国继承法》第二条、第三条、第五条、第十三条之规定，判决如下：'))
with open('/root/data/classify_data/class_test.json', 'r', encoding="utf8") as f:
    ROUGE = np.zeros((4, 3))
    num = 0
    for line in tqdm(f):
        data = json.loads(line)
        id = data.get('id')
        summary = data.get('summary')
        text = data.get('text')
        key_sentences = ''
        for i, item in enumerate(text):
            # print(item['sentence'])
            sentence = item['sentence']
            # pre_label = pred_input(sentence)
            # print(pre_label)
            pre_label = item['label']
            if pre_label == 1:
                key_sentences += sentence
        rouge = evaluate_rouge(key_sentences, summary)
        ROUGE += rouge
        num += 1
        if num > 1000:
            print(ROUGE/1000)
            break
