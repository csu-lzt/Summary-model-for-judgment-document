import json
import numpy as np
from function.str2list import str2list
from function.rouge import Rouge
from tqdm import tqdm


def evaluate_rouge(predict_summary, golden_summary):
    ROUGE = np.zeros((4, 3))
    predict_summary_grams = str2list(predict_summary)
    golden_summary_grams = str2list(golden_summary)
    if predict_summary != '':
        ROUGE = 100 * Rouge(predict_summary_grams, golden_summary_grams)
    # rouge1_F1, rouge1_P, rouge1_R,
    # rouge2_F1, rouge2_P, rouge2_R,
    # rougeL_F1, rougeL_P, rougeL_R,
    # rouge_F1, rouge_P, rouge_R
    # 返回一个4*3的np矩阵
    return ROUGE


if __name__ == '__main__':
    rouge = evaluate_rouge('我说', '我说一个好东西')
    print(rouge)
    rouge = rouge + rouge
    print(rouge)
