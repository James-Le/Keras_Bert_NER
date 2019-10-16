import os
import re
import time
import numpy as np
from tqdm import tqdm_notebook
from keras import backend as K
from utils import *
from models import *
from bert_serving.client import BertClient
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# params
bert_path = "cased_L-12_H-768_A-12"
n_tags = 5
batch_size=32
max_len=10
validation_split=0.2
ip = "localhost"
bc = BertClient(ip)
checkpoint_dir = 'checkpoints/' + str(int(time.time())) + '/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
bst_model_path = checkpoint_dir + 'best.h5'

train_inps, train_tars = [], []

for txt in os.listdir("data"):
    if txt[-4:] == ".txt":
        try:
            with open("data/"+txt) as file:
                tmp_i, tmp_t = [], []
                start = True
                for line in file:
                    if line.strip():
                        if line[:3] == "'''" and start:
                            start = False
                        elif line[:3] == "'''" and not start:
                            start = True
                        elif start:
                            tmp_i.append(line.strip().split("}}")[-1].strip())
                            label = re.findall(r"\d+", line.strip().split("}}")[0])[0]
                            try:
                                label = int(label)
                                tmp_t.append(label)
                            except:
                                print(txt)
                                break
                        else:
                            pass
            train_inps.append(tmp_i)
            train_tars.append(tmp_t)
        except:
            pass

train_inps, test_inps = train_inps[:-10], train_inps[-10:]
train_tars, test_tars = train_tars[:-10], train_tars[-10:]

cut = int(len(train_inps) * validation_split)

train_inps, val_inps = train_inps[:-cut], train_inps[-cut:]
train_tars, val_tars = train_tars[:-cut], train_tars[-cut:]

train_loader = DataGenerator(train_inps, train_tars, batch_size=batch_size, max_len=max_len, ip=ip)
valid_loader = DataGenerator(val_inps, val_tars, batch_size=batch_size, max_len=max_len, ip=ip)

# initialization and training
model = build_model(max_len, n_tags)

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint(bst_model_path, monitor='val_crf_viterbi_accuracy', save_best_only=True, save_weights_only=False)
# tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

model.fit_generator(train_loader,
                    epochs=20,
                    verbose=1,
                    validation_data=valid_loader,
                    callbacks=[early_stopping, model_checkpoint])

X_test = np.zeros((len(test_inps), max_len, 768))

for i,inp in enumerate(test_inps):
    X_test[i,:len(inp),:] = bc.encode(inp)
    
pred = model.predict(X_test)

for i,x in enumerate(test_inps):
    indices = [np.argmax(tok) for tok in pred[i]]
    print(indices[:len(x)])
    print(test_tars[i])
    print(x)