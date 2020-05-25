#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:39:38 2020

@author: antonio

MAIN
"""
from time import time
import numpy as np
import tensorflow as tf
from keras.optimizers import Adagrad
from sklearn.model_selection import StratifiedKFold
import os

from load import load_all
from preprocessing import prepro, weight_computation, get_prepared_data, \
    train_val_splitting
from define_metrics import Metrics_DDI_classification, multi_micro_f1_, \
    get_callbacks_list
from utils import argparser, combine_params
from network_functions import define_network, train_network, get_predictions
from calculate_metrics import get_metrics_cv

if __name__ == '__main__':
    
    in_path, out_path, out_path_figure = argparser()
    #in_path = '/content/gdrive/My Drive/Master/Subjects/TFM/code/DOC/two_steps/second_step/'
    
    ## 1. Load embedding, position matrices and word2int_dict of train set,
    # train and test sets
    emb_matrix,pos_matrix,w2i_dict,VOCAB_SIZE,WV_VECTOR_SIZE,POS_VECTOR_LENGTH,\
        MAX_SENTENCE_LENGTH,X_train,y_train,X_test,y_test = load_all(in_path)
        
    ## 2. Preprocessing
    label2int = {'INT': 0, 'ADVISE': 1, 'EFFECT': 2, 'MECHANISM': 3}
    X_train_i_pad, X_train_p1_pad, X_train_p2_pad, y_train_1h, y_train_i = \
        prepro(X_train,y_train, w2i_dict, label2int, MAX_SENTENCE_LENGTH)
    X_test_i_pad, X_test_p1_pad, X_test_p2_pad, y_test_1h, y_test_i = \
        prepro(X_test,y_test, w2i_dict, label2int, MAX_SENTENCE_LENGTH, 
               test=True, y_trainset=y_train)
    original_train_data = y_train_i, y_train_1h, X_train_i_pad, X_train_p1_pad, X_train_p2_pad
    
    ## 3. Define metrics
    #out_path = "/content/gdrive/My Drive/Master/Subjects/TFM/code/DOC/two_steps/second_step/new/"
    metrics_callbacks = Metrics_DDI_classification()
        
    ## 4. Define network parameters
    fc_size = [64]
    dropout_fc = [0.5]
    oversampling = [False]
    params = [fc_size, dropout_fc, oversampling]
    indexes = ['fc_size' ,'fc_d', 'oversampling']
    network_parameters = combine_params(params, indexes)
    
    ## 5. Define network
    np.random.seed(1)
    tf.random.set_seed(1)
    
    #label2int = {'NONE': 0, 'ADVISE': 1, 'EFFECT': 1, 'INT': 1, 'MECHANISM': 1}
    label2int = {'INT': 0, 'ADVISE': 1, 'EFFECT': 2, 'MECHANISM': 3}
    N_classes = len(set(y_train_i)) # number of output classes in labels vector
    cv_size = 2
    b_size = 50
    n_epoch = 50
    skf = StratifiedKFold(n_splits=cv_size)
    optimizer = Adagrad(lr=0.01)
    metrics = ['acc',  multi_micro_f1_([1,2,3])]
      
    ## 6. Train for all parameters with CV
    # Calculate validation predictions
    for param in network_parameters:
        print('______________________________________________________________')
        print('______________________________________________________________')
        print(param)
        path_arch, model = \
            define_network(out_path, param, emb_matrix, pos_matrix, N_classes, 
                           MAX_SENTENCE_LENGTH, WV_VECTOR_SIZE, VOCAB_SIZE, 
                           POS_VECTOR_LENGTH, optimizer, metrics)
        
        train_data_ready = get_prepared_data(param, original_train_data)

        # Compute class weights
        class_weight_dict = weight_computation(train_data_ready[0], step=2)
        # Ignore INT class
        class_weight_dict[0] = 1
        
        ## Define metric lists
        val_f1_micro = []
        val_precision_micro = []
        val_recall_micro = []
        val_f1_macro = []
        val_precision_macro = []
        val_recall_macro = []
        val_f1_ind = []
        val_precision_ind = []
        val_recall_ind = []
        metrics_list = [val_f1_micro, val_precision_micro, val_recall_micro, 
                        val_f1_macro, val_precision_macro, val_recall_macro, 
                        val_f1_ind, val_precision_ind, val_recall_ind]

  
        ## Stratified Cross-validation
        cv = 0
        for train_idx, val_idx in skf.split(train_data_ready[2], train_data_ready[0]):
            print('----------------------------------------------------------------')
            print(cv)
            
            data_train, data_val = train_val_splitting(train_data_ready, 
                                                       train_idx, val_idx)
            
            # Update checkpoint path
            path_this = out_path + 'models/_fc_size_' + str(param['fc_size']) + \
            '_dropoutfc_' + str(param['fc_d']) + '_cv_' + str(cv) + '_'
              
            callbacks_list = get_callbacks_list(path_this, metrics_callbacks)              
            
            # Reload initial model weights
            model.load_weights(path_arch + 'initial_weights.h5')
          
            # Train
            b_size = 1000
            n_epoch = 5
            t1 = time()
            model = train_network(model, data_train, data_val, class_weight_dict,
                                  callbacks_list, path_this, n_epoch, b_size)
            t2 = time() - t1
            
            # Predictions
            pred, pred_classes = get_predictions(model, data_val)
            metrics_list = get_metrics_cv(pred_classes, data_val[0], path_arch, 
                                          cv, metrics_list)
                
            cv = cv + 1
           
        # Write full report
        with open(path_arch + 'mean_crossval_results.txt', 'w') as f:
          print('---------------------------------------------------------------', file=f)
          print('---------------------------------------------------------------', file=f)
          print('MEAN RESULTS', file=f)
          print('\nMICRO METRICS (excluding INT class)', file=f)
          print('f-score: ', np.mean(metrics_list[0]), file=f)
          print('precision: ', np.mean(metrics_list[1]), file=f)
          print('recall: ', np.mean(metrics_list[2]), file=f) 
        
          print('\nMACRO METRICS (excluding INT class)', file=f)
          print('f-score: ', np.mean(metrics_list[3]), file=f)
          print('precision: ', np.mean(metrics_list[4]), file=f)
          print('recall: ', np.mean(metrics_list[5]), file=f) 
        
          print('\nINDIVIDUAL METRICS', file=f)
          print('f-score: ', str(np.mean(metrics_list[6], axis=0))[1:-1], file=f)
          print('precision: ',str(np.mean(metrics_list[7], axis=0))[1:-1], file=f)
          print('recall: ', str(np.mean(metrics_list[8], axis=0))[1:-1], file=f) 
        
        f.close()
