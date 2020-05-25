#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:34:10 2020

@author: antonio
Network functions
"""
import os
import numpy as np
import tensorflow as tf
import pickle

from keras.models import Sequential
from keras.layers import Embedding, LSTM, MaxPooling1D, Bidirectional, Dense, Flatten, Input, Concatenate, Dropout, GRU
from keras.optimizers import Adagrad
from keras import regularizers
from keras.models import Model
from keras.constraints import maxnorm

def define_network(path_gral, param, emb_matrix, pos_matrix, N_classes,\
                   MAX_SENTENCE_LENGTH, WV_VECTOR_SIZE, VOCAB_SIZE, \
                   POS_VECTOR_LENGTH, _optimizer, _metrics,\
                       _loss='categorical_crossentropy'):
    
    ## Define paths for the results of this architecture
    path_architecture = path_gral + '_fc_size_' + str(param['fc_size']) + \
      '_dropoutfc_' + str(param['fc_d']) + '_'
     
    ## Define architecture
    # 3 Embedding layers: w2v + 2 position vectors
    wv_input = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32', 
                     name='wv_input')
    emb1 = Embedding(output_dim=WV_VECTOR_SIZE, 
                     input_dim=VOCAB_SIZE,
                     input_length=MAX_SENTENCE_LENGTH,
                     weights = [emb_matrix],
                     name='emb1')(wv_input)
    
    pos1_input = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32', 
                       name='pos1_input')
    emb2 = Embedding(output_dim=POS_VECTOR_LENGTH, 
                     input_dim=(MAX_SENTENCE_LENGTH*2) + 1,
                     input_length=MAX_SENTENCE_LENGTH,
                     weights = [pos_matrix],
                     #embeddings_regularizer=regularizers.l2(0.01),
                     name='emb2')(pos1_input)
    
    pos2_input = Input(shape=(MAX_SENTENCE_LENGTH,), dtype='int32', 
                       name='pos2_input')
    emb3 = Embedding(output_dim=POS_VECTOR_LENGTH, 
                     input_dim=(MAX_SENTENCE_LENGTH*2) + 1,
                     input_length=MAX_SENTENCE_LENGTH,
                     weights = [pos_matrix],
                     #embeddings_regularizer=regularizers.l2(0.01),
                     name='emb3')(pos2_input)
    
    # Concatenate embedding layers
    z = Concatenate(name='merged')([emb1, emb2, emb3])
    
    
    # Bi-LSTM
    bilstm = Bidirectional(GRU(units=512, return_sequences=True, dropout=0.5),
                           merge_mode='concat')(z)
    
    
    # Max pooling
    p1 = MaxPooling1D(pool_size=MAX_SENTENCE_LENGTH,
                      strides=MAX_SENTENCE_LENGTH,
                      data_format='channels_last',
                      name='p1')(bilstm)
    
    # Flatten layer
    z = Flatten()(p1)
    
    # FC
    z = Dense(units = param['fc_size'], activation = 'relu')(z)
    
    z = Dropout(param['fc_d'])(z)
    
    # Softmax
    out = Dense(units=N_classes, activation='softmax', name='out')(z)
    
    # Define model
    model = Model(inputs=[wv_input, pos1_input, pos2_input], outputs=[out])
    
    # Compile model and save weights
    model.compile(loss=_loss,
          optimizer=_optimizer,
          metrics=_metrics)
  
    ## Save initial weights
    model.save_weights(path_architecture + 'initial_weights.h5')
    
    return path_architecture, model

def train_network(model, train_data, val_data, class_weight_dict,
                  callbacks_list, out_path, n_epoch=25, b_size=50, _shuffle=True):
    np.random.seed(1)
    tf.random.set_seed(1)

    history = model.fit([train_data[2], train_data[3], train_data[4]],
                        train_data[1],
                        epochs=n_epoch,
                        batch_size=b_size,
                        class_weight=class_weight_dict,
                        shuffle=_shuffle,
                        validation_data=([val_data[2], val_data[3], val_data[4]],
                                         val_data[1]),
                        callbacks=callbacks_list)
    
    # Save the weights
    model.save_weights(out_path+ 'final_weights.h5')
    
    # Save the model architecture
    with open(out_path + 'final_architecture.json', 'w') as f:
        f.write(model.to_json())
    
    #save history
    filehandler = open(out_path + 'history.pkl', 'wb')
    pickle.dump(history, filehandler)
    filehandler.close()


    return model

def get_predictions(model, val_data):
    pred = np.asarray(model.predict([val_data[2], val_data[3], val_data[4]])) # predicted probabilities
    pred_classes = pred.argmax(axis=-1) # predicted classes

    return pred, pred_classes