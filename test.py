# #! -*- coding: utf-8 -*-
# # 测试代码可用性: 提取特征
#
# import numpy as np
# from bert4keras.backend import keras
# from bert4keras.models import build_transformer_model
# from bert4keras.tokenizers import Tokenizer
# from bert4keras.snippets import to_array
#
# config_path = "roberta/bert_config.json"
# checkpoint_path = "roberta/bert_model.ckpt"
# dict_path = "roberta/vocab.txt"
#
# tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
# model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重
#
# # 编码测试
# token_ids, segment_ids = tokenizer.encode(u'语言模型')
# token_ids, segment_ids = to_array([token_ids], [segment_ids])
# print(token_ids)
# print(segment_ids)
# print('\n ===== predicting =====\n')
# print(model.predict([token_ids, segment_ids]))
# """
# 输出：
# [[[-0.63251007  0.2030236   0.07936534 ...  0.49122632 -0.20493352
#     0.2575253 ]
#   [-0.7588351   0.09651865  1.0718756  ... -0.6109694   0.04312154
#     0.03881441]
#   [ 0.5477043  -0.792117    0.44435206 ...  0.42449304  0.41105673
#     0.08222899]
#   [-0.2924238   0.6052722   0.49968526 ...  0.8604137  -0.6533166
#     0.5369075 ]
#   [-0.7473459   0.49431565  0.7185162  ...  0.3848612  -0.74090636
#     0.39056838]
#   [-0.8741375  -0.21650358  1.338839   ...  0.5816864  -0.4373226
#     0.56181806]]]
# """
#
# print('\n ===== reloading and predicting =====\n')
# model.save('test.model')
# del model
# model = keras.models.load_model('test.model')
# print(model.predict([token_ids, segment_ids]))
# import numpy as np
# a=[[1,2],[3,4]]
# b=np.array(a)
# print(a[1][1])
# print(b)
# import json
# from tqdm import tqdm
# from classify_predict.Extractive_model_predict import pred_input

# input_path = "/root/data/abstract_data/用来测试生成模型原始数据-76.json"
# output_path = "/root/data/abstract_data/summary-segment-test.json"

Flag_load = input("load existing model weight?y/n")
print(Flag_load)
if Flag_load == 'y':
    print(111)
print(333)
# if __name__ == "__main__":
#     with open(output_path, 'a', encoding='utf8') as fw:
#         with open(input_path, 'r', encoding="utf8") as f:
#             for line in tqdm(f):
#                 data = json.loads(line)
#                 id = data.get('id')
#                 summary = data.get('summary')
#                 text = data.get('text')  # "text": [{"sentence":"001"},{"sentence":"002"}]
#                 text_pre_label1 = []
#                 text_label1 = []
#                 for i, item in enumerate(text):
#                     sentence = item['sentence']
#                     pre_label = pred_input(sentence)
#                     label = item['label']
#                     if label == 1:
#                         text_label1.append(sentence)
#                     if pre_label == 1:
#                         text_pre_label1.append(sentence)
#                 result = dict(
#                     summary=summary,
#                     label1=text_label1,
#                     pre_label1=text_pre_label1
#                 )
#                 print(text_label1)
#                 print("result:",result)
#                 fw.write(json.dumps(result, ensure_ascii=False) + '\n')
