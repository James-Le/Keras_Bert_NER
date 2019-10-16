from keras import optimizers
from keras.layers import Layer
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy, crf_viterbi_accuracy
from keras.models import Model, Input
from keras.layers import Embedding, Dense, Bidirectional, Dropout, LSTM, TimeDistributed, Masking

# Build model
def build_model(max_para_length, n_tags):
    
    # Bert Embeddings
    bert_output = Input(shape=(max_para_length,768), name="bert_output")
#     input_masks = Input(shape=(max_para_length,), name="input_masks")
    masked = Masking(mask_value=0., input_shape=(max_para_length,768))(bert_output)
    lstm = Bidirectional(LSTM(units=256, return_sequences=True))(masked)
    drop = Dropout(0.1)(lstm)
    dense = TimeDistributed(Dense(256, activation="relu"))(drop)
    crf = CRF(n_tags)
    out = crf(dense)
    model = Model(inputs=bert_output, outputs=out)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#     crf_loss = crf.loss_function()
#     masked_crf_loss = K.sum(crf_loss * input_masks) / K.sum(input_masks)
#     model.add_loss(masked_crf_loss)
    model = Model(inputs=bert_output, outputs=out)
    model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    model.summary()
    
    return model