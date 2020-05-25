#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:04:33 2020

@author: antonio
Calculate metrics
"""
import os
from sklearn.metrics import f1_score, precision_score, recall_score

def compute_metrics(pred_classes, y_val_i_final):

    # MICRO Metrics
    _val_f1_micro = f1_score(y_val_i_final, pred_classes, average='micro', 
                             labels=[1,2,3])
    _val_recall_micro = recall_score(y_val_i_final, pred_classes, 
                                     average='micro', labels=[1,2,3])
    _val_precision_micro = precision_score(y_val_i_final, pred_classes, 
                                           average='micro', labels=[1,2,3])

    # MACRO Metrics
    _val_f1_macro = f1_score(y_val_i_final, pred_classes, average='macro', 
                             labels=[1,2,3])
    _val_recall_macro = recall_score(y_val_i_final, pred_classes, 
                                     average='macro', labels=[1,2,3])
    _val_precision_macro = precision_score(y_val_i_final, pred_classes, 
                                           average='macro', labels=[1,2,3])

    # Metrics per class (esto son vectores!)
    _val_f1_ind = f1_score(y_val_i_final, pred_classes, average=None)
    _val_recall_ind  = recall_score(y_val_i_final, pred_classes, average=None)
    _val_precision_ind  = precision_score(y_val_i_final, pred_classes, average=None)
    
    
    return [_val_f1_micro, _val_recall_micro, _val_precision_micro, _val_f1_macro,
            _val_recall_macro, _val_precision_macro, _val_f1_ind, _val_recall_ind, 
            _val_precision_ind]


def get_metrics_cv(pred_classes, y_val_i_final, path_arch, cv, metrics_list):
    
    # Compute metrics
    _val_f1_micro, _val_recall_micro, _val_precision_micro, _val_f1_macro,\
        _val_recall_macro, _val_precision_macro, _val_f1_ind, _val_recall_ind, \
            _val_precision_ind = compute_metrics(pred_classes, y_val_i_final)
            
    # Write report
    with open(os.path.join(path_arch, 'crossval_results.txt'), 'a') as f:
        print('---------------------------------------------------------------', file=f)
        print(cv, file=f)
        print('\nMICRO METRICS (excluding INT class)', file=f)
        print('f-score: ', _val_f1_micro, file=f)
        print('precision: ', _val_precision_micro, file=f)
        print('recall: ', _val_recall_micro, file=f) 
      
        print('\nMACRO METRICS (excluding INT class)', file=f)
        print('f-score: ', _val_f1_macro, file=f)
        print('precision: ', _val_precision_macro, file=f)
        print('recall: ', _val_recall_macro, file=f) 
      
        print('\nINDIVIDUAL METRICS', file=f)
        print('f-score: ', str(_val_f1_ind)[1:-1], file=f)
        print('precision: ',str(_val_precision_ind)[1:-1], file=f)
        print('recall: ', str(_val_recall_ind)[1:-1], file=f) 
      
        print('\n', file=f)
    
    f.close()
    
    # Append to general metrics
    metrics_list[0].append(_val_f1_micro)
    metrics_list[1].append(_val_precision_micro)
    metrics_list[2].append(_val_recall_micro)
    metrics_list[3].append(_val_f1_macro)
    metrics_list[4].append(_val_precision_macro)
    metrics_list[5].append(_val_recall_macro)
    metrics_list[6].append(_val_f1_ind)
    metrics_list[7].append(_val_precision_ind)
    metrics_list[8].append(_val_recall_ind)
    
    return metrics_list
    
    