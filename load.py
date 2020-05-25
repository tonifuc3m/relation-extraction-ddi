#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:41:17 2020

@author: antonio
Load stuff
"""

import os
import pickle
#from google.colab import drive
#drive.mount('/content/gdrive')

# path_step = "/content/gdrive/My Drive/Master/Subjects/TFM/code/DOC/two_steps/second_step/"

def load_all(path_step,emb_matrix="emb_matrix.pkl",pos_matrix="pos_matrix.pkl",
             w2i_dict="word2int_train.pkl", train="processed_dataset/Xy_train.pkl",
             test="processed_dataset/Xy_test.pkl"):
    '''
    Load embedding, position matrices and word2int dicttionary of train set, 
    train and test sets. Also, compute vocabulary size, word embeddings sizes,
    position vectors sizes and maximum sentence length
    
    Parameters
    ----------
    
    Returns:
    ----------
    emb_matrix: numpy.ndarray
        Embedding matrix (size: vocabulary size x embedding size)
    pos_matrix: numpy.ndarray
        Position embedding matrix (the weights of the position embedding, which
        depend on the word distance to each of the entities). 
        (size: maximum lenght distance*2 + 1 x position vector size)
    w2i_dict: dict
        Dictionary relating every word with an integer. That integer tells us 
        the position in the embedding matrix for the embedding of that word.
    VOCAB_SIZE: int
    WV_VECTOR_SIZE: int
    MAX_SENTENCE_LENGTH: int
    X_train: list
        List with tokenized sentences of train set.
        Every element is a 3 item-touple:  
        (word (string), distance to entity1 (int), distance to entity 2 (int))
    y_train: list
        List with train set labels. Items (class names) are strings.
    X_test: list
    y_test: list
    
    '''
    

    emb_matrix = load_from_pickle_one(path_step, emb_matrix)
    pos_matrix = load_from_pickle_one(path_step, pos_matrix)
    w2i_dict = load_from_pickle_one(path_step, w2i_dict)

    X_train, y_train = load_from_pickle_two(path_step, train)
    X_test, y_test = load_from_pickle_two(path_step, test)

    # Get measures from matrix shapes. To see why, check create_matrices.py
    (VOCAB_SIZE, WV_VECTOR_SIZE) = emb_matrix.shape
    (aux, POS_VECTOR_LENGTH) = pos_matrix.shape
    MAX_SENTENCE_LENGTH = int((aux - 1) / 2)
    
    return emb_matrix,pos_matrix,w2i_dict,VOCAB_SIZE,WV_VECTOR_SIZE,\
        POS_VECTOR_LENGTH,MAX_SENTENCE_LENGTH,X_train,y_train,X_test,y_test

def load_from_pickle_one(path, name):
    '''
    Load object from pickle with ONE object stored
    '''
    infile = open(os.path.join(path, name),'rb')
    loaded_pickle = pickle.load(infile)
    return loaded_pickle

def load_from_pickle_two(path, name):
    '''
    Load object from pickle with TWO object stored
    '''
    infile = open(os.path.join(path, name),'rb')
    loaded_pickle1, loaded_pickle2 = pickle.load(infile)
    return loaded_pickle1, loaded_pickle2

