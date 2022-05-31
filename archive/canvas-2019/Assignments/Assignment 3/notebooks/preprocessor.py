#!/usr/bin/anaconda3/bin/python
from __future__ import absolute_import

import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import string
from PIL import Image
import glob
from time import time
from data_connector import DataConnector
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras import Input, layers
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical


# load doc into memory
def load_doc(params, filename):

    data_path = params['data_path']

    # open the file as read only
    file = open(os.path.join(data_path, filename), 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()

    return text

def load_descriptions(doc):

  
    mapping = dict()
    # process lines
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # extract filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # create the list if needed
        if image_id not in mapping:
            mapping[image_id] = list()
        # store description
        mapping[image_id].append(image_desc)

    print("description before cleaning ...")
    sys.stdout.flush()
    print(mapping['990890291_afc72be141'])
    sys.stdout.flush()

    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in mapping.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            desc = [w.translate(table) for w in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word)>1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc_list[i] =  ' '.join(desc)

    print("description after cleaning ...")
    sys.stdout.flush()
    print(mapping['990890291_afc72be141'])
    sys.stdout.flush()

    conn = DataConnector('./data', 'descriptions.pkl', mapping)
    conn.save_pickle()
    
    return mapping

def load_stored_map_desc():

    conn = DataConnector('./data', 'descriptions.pkl', data=None)
    conn.read_pickle()
    descriptions = conn.read_file


    return descriptions


# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
    # build a list of all description strings
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


# load a pre-defined list of photo identifiers
def load_set(params, filename):

    doc = load_doc(params, filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)

    return set(dataset)


def load_clean_descriptions(descriptions, dataset):

    #descriptions = load_stored_map_desc()

    data_desc = dict()

    i = 0
    for image_id, desc_list in descriptions.items():

      
        if image_id in dataset:
            # create list
            if image_id not in data_desc:
                data_desc[image_id] = list()

            for image_desc in desc_list:
                # wrap description in tokens
                desc = 'startseq ' + image_desc + ' endseq'
                data_desc[image_id].append(desc)


    return data_desc

def preprocess(image_path):

    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = image.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = image.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x


# Function to encode a given image into a vector of size (2048, )
def encode(image, model):

    image = preprocess(image) # preprocess the image
    fea_vec = model.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec

# taken from train description
def vocabulary_indexing(train_descriptions):

    # Create a list of all the training captions
    all_train_captions = []
    for key, val in train_descriptions.items():
        for cap in val:
            all_train_captions.append(cap)

    print("total number of captions: %s" % str(len(all_train_captions)))
    sys.stdout.flush()

    # Consider only words which occur at least 10 times in the corpus
    word_count_threshold = 10
    word_counts = {}
    nsents = 0
    for sent in all_train_captions:
        nsents += 1
        for w in sent.split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))
    sys.stdout.flush()

    ixtoword = {}
    wordtoix = {}

    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    conn = DataConnector('./data', 'wordtoix.pkl', wordtoix)
    conn.save_pickle()

    conn = DataConnector('./data', 'ixtoword.pkl', ixtoword)
    conn.save_pickle()

    vocab_size = len(ixtoword) + 1 # one for appended 0's
    print("vocab_size after preprocessing: %s" % vocab_size)
    sys.stdout.flush()

    return 0


# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    
    return all_desc

# calculate the length of the description with the most words
def max_length(descriptions):
    lines = to_lines(descriptions)
    
    return max(len(d.split()) for d in lines)


def load_glove_embedding(params, wordtoix):

    data_path = params['data_path']
    vocab_size = len(wordtoix) + 1

    embeddings_index = {} # empty dictionary
    f = open(os.path.join(data_path, 'glove.6B.300d.txt'), encoding="utf-8")

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_dim = 300

    # Get 300-dim dense vector for each of the 10000 words in out vocabulary
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in wordtoix.items():
        #if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            embedding_matrix[i] = embedding_vector

    embedding_matrix = np.array(embedding_matrix)

    conn = DataConnector('./data', 'embedding_matrix', embedding_matrix)
    conn.save_numpys()

    return embedding_matrix


def sequence_preprocessing(dataset, descriptions, features, wordtoix, max_length):

    img_in = []
    cap_in = []
    cap_out = []

    for key, desc_list in descriptions.items():
      
      img_vect = features[key+'.jpg']
      
      for desc in desc_list:

        seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]


        seq_in = seq[0:(len(seq)-1)]
        seq_out = seq[1:]
        
        seq_in = sequence.pad_sequences([seq_in], padding='post', maxlen=max_length)[0]
        seq_out = sequence.pad_sequences([seq_out], padding='post', maxlen=max_length)[0]
        
        
        img_in.append(img_vect)
        cap_in.append(seq_in)
        cap_out.append(seq_out)

    img_in = np.array(img_in)
    cap_in = np.array(cap_in)
    cap_out = np.array(cap_out)

    conn = DataConnector('./data', '%s_img_in'%dataset, img_in)
    conn.save_numpys()

    conn = DataConnector('./data', '%s_cap_in'%dataset, cap_in)
    conn.save_numpys()

    conn = DataConnector('./data', '%s_cap_out'%dataset, cap_out)
    conn.save_numpys()

    return img_in, cap_in, cap_out

