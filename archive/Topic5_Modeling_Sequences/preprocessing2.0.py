#!/usr/bin/anaconda3/bin/python
from __future__ import absolute_import

import os
import sys
sys.path.append(os.getcwd())
from preprocessor import *


config = {
    
    'data_path': '/home/datasets/recsys',
    'preps': './data'
}

if __name__ == '__main__':

    conn = DataConnector('./data', 'wordtoix.pkl', data=None)
    conn.read_pickle()
    wordtoix = conn.read_file

    conn = DataConnector('./data', 'ixtoword.pkl', data=None)
    conn.read_pickle()
    ixtoword = conn.read_file

    '''

    embedding_matrix = load_glove_embedding(config, wordtoix)

    '''

    conn = DataConnector('./data', 'train_descriptions.pkl', data=None)
    conn.read_pickle()
    train_descriptions = conn.read_file

    conn = DataConnector('./data', 'dev_descriptions.pkl', data=None)
    conn.read_pickle()
    dev_descriptions = conn.read_file

    conn = DataConnector('./data', 'test_descriptions.pkl', data=None)
    conn.read_pickle()
    test_descriptions = conn.read_file

    conn = DataConnector('./data', 'encoding_train.pkl', data=None)
    conn.read_pickle()
    encoding_train = conn.read_file

    conn = DataConnector('./data', 'encoding_dev.pkl', data=None)
    conn.read_pickle()
    encoding_dev = conn.read_file

    conn = DataConnector('./data', 'encoding_test.pkl', data=None)
    conn.read_pickle()
    encoding_test = conn.read_file


    ### sequence preprocessing for training set
    train_img_in, train_cap_in, train_cap_out = sequence_preprocessing('train', train_descriptions, encoding_train, wordtoix, 34)

    print("train img_in shape: %s"% str(train_img_in.shape))
    print("train cap_in shape: %s"% str(train_cap_in.shape))
    print("train cap_out shape: %s"% str(train_cap_out.shape))

    ### sequence preprocessing for dev set
    dev_img_in, dev_cap_in, dev_cap_out = sequence_preprocessing('dev', dev_descriptions, encoding_dev, wordtoix, 34)

    print("dev img_in shape: %s"% str(dev_img_in.shape))
    print("dev_cap_in shape: %s"% str(dev_cap_in.shape))
    print("dev_cap_out shape: %s"% str(dev_cap_out.shape))

    ### sequence preprocessing for test set
    test_img_in, test_cap_in, test_cap_out = sequence_preprocessing('test', test_descriptions, encoding_test, wordtoix, 34)

    print("test_img_in shape: %s"% str(test_img_in.shape))
    print("test_cap_in shape: %s"% str(test_cap_in.shape))
    print("test_cap_out shape: %s"% str(test_cap_out.shape))

