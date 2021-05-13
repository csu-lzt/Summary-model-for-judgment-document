import re


def simplify(text, i):
    sim_text = text.replace(u'\u3000', u'')
    if i == 0:
        if re.search(r"继承纠纷", text):
            sim_text = r"本案为继承纠纷。"
        if re.search(r"借款合同", text):
            sim_text = r"原被告系借款合同关系。"
        if re.search(r"租赁合同", text):
            sim_text = r"原被告系租赁合同关系。"
        if re.search(r"侵权责任", text):
            sim_text = r"原被告系侵权责任纠纷。"
        if re.search(r"劳动合同", text):
            sim_text = r"原被告系劳动合同关系。"
    else:
        sim_text = re.sub(r'[\(|（|【].*?[\)|）|】]', "", text)
    return sim_text

def split_part(text, maxlen):  # text为文本列表；maxlen每部分最大长度；分成2/3/4部分
    text_str = ''
    part_list = []
    text_n = len(text)
    for i in range(text_n):
        text[i] = simplify(text[i], i)
        strlen = len(text_str)
        ilen = len(text[i])
        if (strlen + ilen) < maxlen:
            text_str += text[i]
        else:
            part_list.append(text_str)
            text_str = text[i]
        if i == text_n - 1:
            part_list.append(text_str)
    return part_list