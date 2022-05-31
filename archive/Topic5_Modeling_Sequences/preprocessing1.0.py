#!/usr/bin/anaconda3/bin/python
from __future__ import absolute_import

import os
import sys
sys.path.append(os.getcwd())
from preprocessor import *


config = {
    
    'data_path': '/home/datasets/recsys',
}

if __name__ == '__main__':


    doc = load_doc(config, 'Flickr8k.token.txt')
    print(doc[:300])
    sys.stdout.flush()

    descriptions = load_descriptions(doc)
    print('Loaded: %d ' % len(descriptions))
    sys.stdout.flush()

    # summarize vocabulary
    vocabulary = to_vocabulary(descriptions)
    print('Original Vocabulary Size (before preprocessing): %d' % len(vocabulary))
    sys.stdout.flush()

    train = load_set(config, 'Flickr_8k.trainImages.txt')
    print('Dataset: %d' % len(train))
    sys.stdout.flush()

    # descriptions of training set
    train_descriptions = load_clean_descriptions(descriptions, train)
    print('Descriptions: train=%d' % len(train_descriptions))
    sys.stdout.flush()
    # save train - desc
    conn = DataConnector('./data', 'train_descriptions.pkl', train_descriptions)
    conn.save_pickle()

    # determine the maximum sequence length
    max_length = max_length(train_descriptions)
    print('Description Length: %d' % max_length)
    sys.stdout.flush()

    ### Vocabulary indexing
    vocabulary_indexing(train_descriptions)

    # load validation dataset (1K)
    dev = load_set(config, 'Flickr_8k.devImages.txt')
    print('Validation Dataset: %d' % len(dev))
    sys.stdout.flush()

    # descriptions of development set
    dev_descriptions = load_clean_descriptions(descriptions, dev)
    print('Descriptions: validation=%d' % len(dev_descriptions))
    sys.stdout.flush()

    # save dev - desc
    conn = DataConnector('./data', 'dev_descriptions.pkl', dev_descriptions)
    conn.save_pickle()

    # load test dataset (1K)
    test = load_set(config, 'Flickr_8k.testImages.txt')
    print('Test Dataset: %d' % len(test))
    sys.stdout.flush()

    # descriptions of test set
    test_descriptions = load_clean_descriptions(descriptions, test)
    print('Descriptions: test=%d' % len(test_descriptions))
    sys.stdout.flush()

    # save test - desc
    conn = DataConnector('./data', 'test_descriptions.pkl', test_descriptions)
    conn.save_pickle()


    # Below path contains all the images
    images_path = '/home/datasets/recsys/Flicker8k_Dataset/'
    # Create a list of all image names in the directory
    img = glob.glob(images_path + '*.jpg')

    # Below file conatains the names of images to be used in train data
    train_images_file = '/home/datasets/recsys/Flickr_8k.trainImages.txt'
    # Read the train image names in a set
    train_images = set(open(train_images_file, 'r').read().strip().split('\n'))

    # Create a list of all the training images with their full path names
    train_img = []

    for i in img: # img is list of full path names of all images
        if i[len(images_path):] in train_images: # Check if the image belongs to training set
            train_img.append(i) # Add it to the list of train images


    # Below file conatains the names of images to be used in validation data
    dev_images_file = '/home/datasets/recsys/Flickr_8k.devImages.txt'
    # Read the validation image names in a set# Read the test image names in a set
    dev_images = set(open(dev_images_file, 'r').read().strip().split('\n'))

    # Create a list of all the test images with their full path names
    dev_img = []

    for i in img: # img is list of full path names of all images
        if i[len(images_path):] in dev_images: # Check if the image belongs to test set
            dev_img.append(i) # Add it to the list of test images


    # Below file conatains the names of images to be used in test data
    test_images_file = '/home/datasets/recsys/Flickr_8k.testImages.txt'
    # Read the validation image names in a set# Read the test image names in a set
    test_images = set(open(test_images_file, 'r').read().strip().split('\n'))

    # Create a list of all the test images with their full path names
    test_img = []

    for i in img: # img is list of full path names of all images
        if i[len(images_path):] in test_images: # Check if the image belongs to test set
            test_img.append(i) # Add it to the list of test images


    # Load the inception v3 model
    model = InceptionV3(weights='imagenet')
    # Create a new model, by removing the last layer (output layer) from the inception v3
    model_new = Model(model.input, model.layers[-2].output)


    start = time()
    encoding_train = {}
    for img in train_img:
        encoding_train[img[len(images_path):]] = encode(img, model_new)
    print("Time taken for encoding training set (in seconds) =", time()-start)
    sys.stdout.flush()

    # Save the bottleneck train features to disk
    conn = DataConnector('./data', 'encoding_train.pkl', encoding_train)
    conn.save_pickle()

    start = time()
    encoding_dev = {}
    for img in dev_img:
        encoding_dev[img[len(images_path):]] = encode(img, model_new)
    print("Time taken for encoding dev set (in seconds) =", time()-start)


    # Save the bottleneck dev features to disk
    conn = DataConnector('./data', 'encoding_dev.pkl', encoding_dev)
    conn.save_pickle()

    start = time()
    encoding_test = {}
    for img in test_img:
        encoding_test[img[len(images_path):]] = encode(img, model_new)
    print("Time taken for encoding test set (in seconds) =", time()-start)


    # Save the bottleneck test features to disk
    conn = DataConnector('./data', 'encoding_test.pkl', encoding_test)
    conn.save_pickle()

   