import json
import re
import numpy as np
from Evaluate_rouge import evaluate_rouge
from tqdm import tqdm
from TextRank import textRank_extract

# print(pred_input('综上，根据《中华人民共和国继承法》第二条、第三条、第五条、第十三条之规定，判决如下：'))
with open('/root/data/classify_data/class_test.json', 'r', encoding="utf8") as f:
    ROUGE = np.zeros((4, 3))
    num = 0
    for line in tqdm(f):
        data = json.loads(line)
        id = data.get('id')
        summary = data.get('summary')
        text = data.get('text')
        key_sentences = textRank_extract(text)

        rouge = evaluate_rouge(key_sentences, summary)
        ROUGE += rouge
        num += 1
        if num > 1000:
            print(ROUGE / 1000)
            break

#Baseline
        # for i, item in enumerate(text):
        #     sentence = item['sentence']
        #     if re.search(r"诉讼请求：", sentence):
        #         text0 = text[i]["sentence"]
        #         text1 = text[i + 1]["sentence"]
        #         text2 = text[i + 2]["sentence"]
        #         break
        #     else:
        #         text0 = text[11]["sentence"]
        #         text1 = text[12]["sentence"]
        #         text2 = text[13]["sentence"]
        # key_sentences = text0 + text1 + text2