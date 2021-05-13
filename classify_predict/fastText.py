import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from keras_textclassification.text_classification_api import train
train(graph='TextRNN',
      label=2,
      path_train_data='/root/data/class_data/class_train.csv',path_dev_data='/root/data/class_data/class_test.csv', rate=1)
