import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from bert.tokenization import FullTokenizer
from tqdm import tqdm_notebook
from keras import backend as K
from utils import *
from models import *

# Initialize session
sess = tf.Session()

# Params for bert model and tokenization
bert_path = "cased_L-12_H-768_A-12"
max_seq_length = 50

train_words, train_tags = [], []

with open() as file:
    for line in file:
        train_words.append(line.strip().split())

with open() as file:
    for line in file:
        train_tags.append(line.strip().split())

tags = set()
for ts in train_tags:
    for i in ts:
        tags.add(i)

tag2idx = {t: i+1 for i, t in enumerate(list(tags))}
tag2idx["-PAD-"] = 0
n_tags = len(tag2idx)

train_tag_ids = [list(map(lambda x: tag2idx[x], sample)) for sample in train_tags]

# Instantiate tokenizer
tokenizer = create_tokenizer_from_hub_module()

# Convert data to InputExample format
train_examples = convert_text_to_examples(train_words, train_tag_ids)

# Convert to features
(train_input_ids, train_input_masks, train_segment_ids, train_tag_ids 
) = convert_examples_to_features(tokenizer, train_examples, max_seq_length=max_seq_length)

# initialization and training
model = build_model(max_seq_length)

history = model.fit(
    [train_input_ids, train_input_masks, train_segment_ids], 
    train_tag_ids,
    validation_split=0.2,
    epochs=10,
    batch_size=64,
)