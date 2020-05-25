#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:24:05 2020

@author: antonio
Metrics functions
"""
import os
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, LearningRateScheduler


# from https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras

def binary_f1_(y_true, y_pred):
  """F1 metric.
  
  Only computes a batch-wise average of f1 score.
  
  This metric is only computed for the POSITIVE classes. 
  If the label of a class is zero, this metric ignores 
  its results.
  
  Put INT sentences as zero to ignore them in the metric
  computation.
  """
  def recall(y_true, y_pred):
      """Recall metric.

      Only computes a batch-wise average of recall.

      Computes the recall, a metric for multi-label classification of
      how many relevant items are selected.
      """
      true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
      possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
      recall = true_positives / (possible_positives + K.epsilon())
      return recall

  def precision(y_true, y_pred):
      """Precision metric.

      Only computes a batch-wise average of precision.

      Computes the precision, a metric for multi-label classification of
      how many selected items are relevant.
      """
      true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
      predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
      precision = true_positives / (predicted_positives + K.epsilon())
      return precision
  precision = precision(y_true, y_pred)
  recall = recall(y_true, y_pred)
  return 2*((precision*recall)/(precision+recall+K.epsilon()))

def multi_micro_f1_(list_of_labels):
  """F1 metric.
  
  Only computes a batch-wise average of f1 score.
  
  """
  def micro_f1(y_true, y_pred):
    y_true_i = K.argmax(y_true, axis=-1)
    y_preds_i = K.argmax(y_pred, axis=-1)
  
    TP = K.zeros(shape=(1), dtype='float32')
    Possible_P = K.zeros(shape=(1), dtype='float32')
    Pred_P = K.zeros(shape=(1), dtype='float32')

    for label in list_of_labels:
      mask = K.cast(K.equal(y_true_i, label), 'float32')
      true_positives = K.sum(K.cast(K.equal(y_true_i, y_preds_i), 
                                    'float32') * mask, 
                             keepdims=True)
      possible_positives = K.sum(mask, keepdims=True)
      predicted_positives = K.sum(K.cast(K.equal(y_preds_i, label), 
                                         'float32'),
                                  keepdims=True)

      TP = K.concatenate([TP, true_positives])
      Possible_P = K.concatenate([Possible_P, possible_positives])
      Pred_P = K.concatenate([Pred_P, predicted_positives])  

    recall = K.sum(TP) / (K.sum(Possible_P) + K.epsilon())
    precision = K.sum(TP) / (K.sum(Pred_P) + K.epsilon())
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
  
  return micro_f1


# Callbacks
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

class Metrics_DDI_detection(Callback):
  def on_train_begin(self, logs = {}):
    self.val_f1s = []
    self.val_recalls = []
    self.val_precisions = []
    self.val_accuracies = []   

  
  def on_epoch_end(self, batch, logs):
    val_w = self.validation_data[0]
    val_p1 = self.validation_data[1]
    val_p2 = self.validation_data[2]
    
    val_predict = np.asarray(self.model.predict([val_w, val_p1, val_p2])) # predicted probabilities
    val_predict_classes = val_predict.argmax(axis=-1) # predicted classes
    
    val_targ = self.validation_data[3] # 1hot encoded labels
    val_targ_formatted = [np.where(r==1)[0][0] for r in val_targ] # integer format labels
    
    # Metrics of positive class
    _val_f1 = f1_score(val_targ_formatted, val_predict_classes, average=None)[1]
    _val_recall = recall_score(val_targ_formatted, val_predict_classes, average=None)[1]
    _val_precision = precision_score(val_targ_formatted, val_predict_classes, average=None)[1]
    _val_accuracy = accuracy_score(val_targ_formatted, val_predict_classes)
    
    self.val_f1s.append(_val_f1)
    self.val_recalls.append(_val_recall)
    self.val_precisions.append(_val_precision)
    self.val_accuracies.append(_val_accuracy)
    
    
    print(' Positive Class (DDI): — custom_val_f1: %.4f — val_precision: %.4f — val_recall %.4f' 
          % (_val_f1, _val_precision, _val_recall))
    
    print(' - custom_val_accuracy %.4f' % _val_accuracy)
    
    return
  
  
class Metrics_DDI_classification(Callback):
  def on_train_begin(self, logs = {}):
    self.val_f1s_micro = []
    self.val_recalls_micro = []
    self.val_precisions_micro = []
    self.val_f1s_macro = []
    self.val_recalls_macro = []
    self.val_precisions_macro = []
    self.val_f1s_ind = []
    self.val_recalls_ind = []
    self.val_precisions_ind = []
    
    self.val_accuracies = []   

  
  def on_epoch_end(self, batch, logs):
    val_w = self.validation_data[0]
    val_p1 = self.validation_data[1]
    val_p2 = self.validation_data[2]
    
    val_predict = np.asarray(self.model.predict([val_w, val_p1, val_p2])) # predicted probabilities
    val_predict_classes = val_predict.argmax(axis=-1) # predicted classes
    
    val_targ = self.validation_data[3] # 1hot encoded labels
    val_targ_formatted = [np.where(r==1)[0][0] for r in val_targ] # integer format labels
    
    # MICRO Metrics
    _val_f1_micro = f1_score(val_targ_formatted, 
                             val_predict_classes,
                             average='micro', labels=[1,2,3])
    _val_recall_micro = recall_score(val_targ_formatted, 
                                     val_predict_classes, 
                                     average='micro', labels=[1,2,3])
    _val_precision_micro = precision_score(val_targ_formatted,
                                           val_predict_classes, 
                                           average='micro', labels=[1,2,3])
    
    self.val_f1s_micro.append(_val_f1_micro)
    self.val_recalls_micro.append(_val_recall_micro)
    self.val_precisions_micro.append(_val_precision_micro)
    
    # MACRO Metrics
    _val_f1_macro = f1_score(val_targ_formatted, 
                             val_predict_classes, 
                             average='macro', labels=[1,2,3])
    _val_recall_macro = recall_score(val_targ_formatted, 
                                     val_predict_classes, 
                                     average='macro', labels=[1,2,3])
    _val_precision_macro = precision_score(val_targ_formatted, 
                                           val_predict_classes, 
                                           average='macro', labels=[1,2,3])
        
    self.val_f1s_macro.append(_val_f1_macro)
    self.val_recalls_macro.append(_val_recall_macro)
    self.val_precisions_macro.append(_val_precision_macro)
    
    # Metrics per class (esto son vectores!)
    _val_f1_ind = f1_score(val_targ_formatted, 
                           val_predict_classes,
                           average=None)
    _val_recall_ind  = recall_score(val_targ_formatted, 
                                    val_predict_classes, 
                                    average=None)
    _val_precision_ind  = precision_score(val_targ_formatted, 
                                          val_predict_classes, 
                                          average=None)
    
    self.val_f1s_ind.append(_val_f1_ind)
    self.val_recalls_ind.append(_val_recall_ind)
    self.val_precisions_ind.append(_val_precision_ind)
    
    # Accuracy
    _val_accuracy = accuracy_score(val_targ_formatted, val_predict_classes)
    
    self.val_accuracies.append(_val_accuracy)  
    
    # Prints
    print(' MICRO Metrics: — custom_val_f1: %.4f — val_precision: %.4f — val_recall: %.4f' 
          % (_val_f1_micro, _val_precision_micro, _val_recall_micro))

    print(' MACRO Metrics: — custom_val_f1: %.4f — val_precision: %.4f — val_recall: %.4f' 
          % (_val_f1_macro, _val_precision_macro, _val_recall_macro))

    print(' Individual Metrics: — custom_val_f1: ' + str(_val_f1_ind)[1:-1] +
          '— val_precision: ' + str(_val_precision_ind)[1:-1] + 
          '— val_recall: ' + str(_val_recall_ind)[1:-1])

    print(' - custom_val_accuracy %.4f' % _val_accuracy)
    
    return


def get_callbacks_list(path, metrics=Metrics_DDI_classification()):
    
    csvlogger = CSVLogger(path + 'wModel-baseline-log.csv',
                          separator=',', append=False)
    earlystopping = EarlyStopping(monitor='val_micro_f1', patience=5, mode='max',
                                  restore_best_weights=True) # according to positive f1 (defined in 'metrics' parameter of compile)
    
    callbacks_list = [csvlogger, earlystopping, metrics]
    
    return callbacks_list
