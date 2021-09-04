# https://www.tensorflow.org/tutorials/keras/text_classification


import os
import re
import shutil
import string

import matplotlib.pyplot as plt
import tensorflow as tf


url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
dataset = tf.keras.utils.get_file('aclImdb_v1', url, untar=True, cache_dir='.', cache_subdir='')
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
train_dir = os.path.join(dataset_dir, 'train')

sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
    print(f.read())
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)