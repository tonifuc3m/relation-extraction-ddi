#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:52:18 2020

@author: antonio
Preprocess functions
"""

from sklearn.preprocessing import OneHotEncoder
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def custom_pad(X, MAX_SENTENCE_LENGTH):
    '''
    Add padding to X (add zeros until all sentences have same length).
    
    Parameters
    ----------
    X: list
        List of lists. Every item is a list of integers (every integer
        corresponds to a word).
    MAX_SENTENCE_LENGTH: int
        Maximum length of a sentence (we will pad sentences until this number)
        
    Returns
    ----------
    X_padded: list
        List of lists. Every item is a list of integers (every integer 
        corresponds to a word). All lists have same length.
    
    '''
    X_padded = pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH, padding="post",
                             truncating="post", value = 0)
    return X_padded

def word2int(train_words, word_index):
    '''
    Translate words to integers
    
    Parameters
    ----------
    train_words: list
        List of lists. Every item is a tokenized sentence (a list of words)
    w2i_dict: word_index
        Dictionary relating every word with an integer. That integer tells us 
        the position in the embedding matrix for the embedding of that word.
    
    Returns
    ----------
    train: list
        List of lists. Every item is a list of integers (every integer
        corresponds to a word)
    
    '''
    train = [[word_index.get(t, 0) for t in sentence] for sentence in train_words]
    return train

def onehotenc(y, test=False, y_trainset=None):
    '''
    One-hot encoding list of integers.
    
    Parameters
    ----------
    y: list
        List of integers ([1,2,1,3,...])
    test: bool
        Whether we are one-hot encoding the test set.
    y_trainset: list
        Required if test=True. Y labels of trainset.
    
    Returns
    ----------
    y_1h: list
        List of lists([[1,0,0], [0,1,0], [1,0,0], [0,0,1],...])
    y: list
        Same as input (why am I returning this?...).
    
    '''
    onehotencoder = OneHotEncoder()
    if test==False:
        y_1h = onehotencoder.fit_transform(np.array(y).reshape(-1, 1)).toarray()
    else:
        try:
            onehotencoder.fit(np.array(y_trainset).reshape(-1, 1))
            y_1h = onehotencoder.transform(np.array(y).reshape(-1, 1)).toarray()
        except:
            print('If test=True, need to provide a y_trainset set to fit the OneHotEncoder')
            y_1h=None
      
    return y_1h, y
      

def prepro(X, y, w2i_dict, label2int, MAX_SENTENCE_LENGTH, test=False, 
           y_trainset=None):
    '''
    Wrapper of all preprocessing: 
        1. separate words from position vectors (from X numpy arrays)
        2. map words (of sentences) to integers
        3. map y labels to integers
        4. one-hot encoding labels
        5. padding
    
    Parameters
    ----------
    X: list
        List with tokenized sentences of set.
        Every element is a 3 item-touple:  
        (word (string), distance to entity1 (int), distance to entity 2 (int))
    y: list
        List with set labels. Items (class names) are strings.
    w2i_dict: dict
        Dictionary relating every word with an integer. That integer tells us 
        the position in the embedding matrix for the embedding of that word.
    label2int: dict
        Dictionary relating every class with an integer.
    test: boolean
        Indicates whether we are preprocessing the training set. If true, this 
        requires y_trainset. Test set is preprocesed
    y_trainset: list
        List with y labels of train set
        
    Returns
    ----------
    X_i_prepro: list
        List of lists. Every item is a list of integers (every integer 
        corresponds to a word). All lists have same length.
    X_p1_prepro: list
        List of lists. Every item is a list of integers (every integer 
        corresponds to the distance from the word to the entity 1). 
        All lists have same length.
    X_p2_prepro: list
        List of lists. Every item is a list of integers (every integer 
        corresponds to the distance from the word to the entity 2).
        All lists have same length.
    y_1h: list
        One-hot encoded labels. 
        List of lists([[1,0,0], [0,1,0], [1,0,0], [0,0,1],...])
    y_i: list
        List of labels (integers) ([1,2,1,3,...])
    
    '''
    # Separate words from position vectors
    X_w = list(map(lambda x: [a[0] for a in x], X))
    X_p1 = list(map(lambda x: [a[1] for a in x], X))
    X_p2 = list(map(lambda x: [a[2] for a in x], X))
    
    # word2int 
    X_i = word2int(X_w, w2i_dict)
    
    # label2int
    y_i = [label2int[i] for i in y]
    if test==True:
      y_trainset = [label2int[i] for i in y_trainset]
    
    # One-Hot Encoding labels
    y_1h, y_i = onehotenc(y_i, test=test, y_trainset=y_trainset)
        
    # Padding
    X_i_prepro = custom_pad(X_i, MAX_SENTENCE_LENGTH)
    X_p1_prepro = custom_pad(X_p1, MAX_SENTENCE_LENGTH)
    X_p2_prepro = custom_pad(X_p2, MAX_SENTENCE_LENGTH) 
    
    return X_i_prepro, X_p1_prepro, X_p2_prepro, y_1h, y_i


def oversampling(X_i, X_p1, X_p2, y_i, strategy='auto', ratio=None, sampling_type='svm'): 
    '''
    ratio = # majority class instances / # minority class instances
    
    Parameters
    ----------
    
    Returns
    ----------
    
    '''
    # Initialize and join 3 matrices
    _,ei = X_i.shape
    _,e1 = X_p1.shape
    _,e2 = X_p2.shape
    X_total = np.concatenate((X_i, X_p1, X_p2), axis = 1)
    
    # Over sampling
    if sampling_type=='svm':
        from imblearn.over_sampling import SVMSMOTE
      
        if ratio == None:
            os = SVMSMOTE(random_state=0)
        else:
            os = SVMSMOTE(sampling_strategy=1/ratio, random_state=0)
    else:
        from imblearn.over_sampling import RandomOverSampler
        
        if ratio == None:
            os = RandomOverSampler(random_state=0)
        else:
            os = RandomOverSampler(sampling_strategy=1/ratio, random_state=0)
    
    X_res, y_res = os.fit_sample(X_total, y_i)
    
    # Separate in 3 matrices
    X_i_res = X_res[:,0:ei]
    X_p1_res = X_res[:,ei:ei+e1]
    X_p2_res = X_res[:,ei+e1:]
    
    return X_i_res, X_p1_res, X_p2_res, y_res, X_res
  
# 3.7 Calculate weights depending on how present each category is
def weight_computation(y=None, ratio=None, step=1): 
    '''
    Compute weights to include in the loss function according to class proportions or to pre-defined
    ratio.
    Only works for binary classification problems.
    
    ratio = # negative class instances / # positive class instances
    
    Parameters
    ----------
    step: int
        1 for first stage, 2 for second stage
    
    Returns
    ----------
    
    '''
    if ratio==None: 
        _, counts_train = np.unique(np.array(y), return_counts=True)
      
        total_cases = sum(counts_train)
        class_weight = [1 / (c/total_cases) for c in counts_train]
        if step==1:
            class_weight_dict = {0:class_weight[0], 1: class_weight[1]}
        else: 
            class_weight_dict = {0:class_weight[0], 1: class_weight[1], 
                                2:class_weight[2], 3: class_weight[3]}
    else: 
        if step==1:
            class_weight_dict = {0:1.0, 1: float(ratio)}
        else: 
            class_weight_dict = {0:class_weight[0], 1: class_weight[1], 
                          2:class_weight[2], 3: class_weight[3]}
      
    return class_weight_dict

def print_class_weights(y_train, y_test, out_path):
    
    import matplotlib.pyplot as plt
    import seaborn as sns 
    categories_train, counts_train = np.unique(np.array(y_train), return_counts=True)
    categories_test, counts_test = np.unique(np.array(y_test), return_counts=True)
    
    fig = plt.figure(figsize=(15,8))
    (ax_train, ax_test) = fig.subplots(ncols=2, nrows=1)
    g1 = sns.barplot(x=counts_train, y=categories_train, ax=ax_train)
    ax_train.set_xlabel('Number of sentences')
    g2 = sns.barplot(x=counts_test, y=categories_test, ax=ax_test)
    ax_test.set_xlabel('Number of sentences')
    g1.set_title("Category distribution on training dataset")
    g2.set_title("Category distribution on testing dataset")
    fig.savefig(out_path)
    
    
    
def get_prepared_data(param, data_list):
    '''
    Return data prepared for training

    Parameters
    ----------
    param : dict
        output from utils.combine_params.
    data_list : list
        List with data: 
            y_train_i = data_list[0]
            y_train_1h = data_list[1]
            X_train_i_pad = data_list[2]
            X_train_p1_pad = data_list[3]
            X_train_p2_pad = data_list[4]

    Returns
    -------
    : list
        y_train_i_prepared : TYPE
            DESCRIPTION.
        y_train_1h_prepared : TYPE
            DESCRIPTION.
        X_train_i_prepared : TYPE
            DESCRIPTION.
        X_train_p1_prepared : TYPE
            DESCRIPTION.
        X_train_p2_prepared : TYPE
            DESCRIPTION.

    '''
    
    if param['oversampling'] !=False:
        # Oversampling
        X_train_i_res, X_train_p1_res, X_train_p2_res, y_res, X_res = \
                oversampling(data_list[2], data_list[3], data_list[4], data_list[0],
                             strategy=param['overs_strat'], ratio=param['overs_ratio'], 
                             sampling_type=param['overs_type'])
        y_train_res_1h, y_res = onehotenc(y_res, test=False, y_trainset=None)
        return [y_res, y_train_res_1h, X_train_i_res, X_train_p1_res, X_train_p2_res]

    return data_list
    
            
def train_val_splitting(data_prepared, train_idx, val_idx): 
    '''
    Divide training data into train and validation

    Parameters
    ----------
    data_prepared : TYPE
        DESCRIPTION.
    train_idx : TYPE
        DESCRIPTION.
    val_idx : TYPE
        DESCRIPTION.

    Returns
    -------
    train_data : TYPE
        DESCRIPTION.
    val_data : TYPE
        DESCRIPTION.

    '''
    
    # Validation and Train Split
    y_train_i_final = list(np.array(data_prepared[0])[train_idx])
    y_train_final = data_prepared[1][train_idx,:]
    X_train_i_final = data_prepared[2][train_idx,:]
    X_train_p1_final = data_prepared[3][train_idx,:]
    X_train_p2_final = data_prepared[4][train_idx,:]
    train_data = [y_train_i_final, y_train_final, X_train_i_final, 
                           X_train_p1_final, X_train_p2_final]

    y_val_i_final = list(np.array(data_prepared[0])[val_idx])
    y_val_final = data_prepared[1][val_idx,:]
    X_val_i_final = data_prepared[2][val_idx,:]
    X_val_p1_final = data_prepared[3][val_idx,:]
    X_val_p2_final = data_prepared[4][val_idx,:]
    val_data = [y_val_i_final, y_val_final, X_val_i_final, X_val_p1_final, 
                X_val_p2_final]

    
    return train_data, val_data
    