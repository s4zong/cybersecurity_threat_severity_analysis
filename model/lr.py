
"""
train simple LR classifier
"""

import numpy as np
from collections import Counter

import pickle
import json

from sklearn.linear_model import LogisticRegression
from scipy.sparse import *


def convertToSparseMatrix(features_idx, features_dict):

    """
    convert feature idx matrix in to sparse matrix
    :param features_idx: [list] original feature idx matrix
    :param features_dict: [dict] train_ngram_dict (for determine the dimension)
    :return: [dict] sparse matrix tuple
    """

    ## generate 1 locations for sparse matrix
    location = []
    for idx, line in enumerate(features_idx):
        for token in line:
            each_location = []
            each_location.append(idx)
            each_location.append(token)
            location.append(each_location)

    ## row
    row = [i[0] for i in location]
    col = [i[1] for i in location]

    ## generate 1 s
    elements = [int(i) for i in list(np.ones((len(location))))]

    ## dim for original matrix
    dim = [len(features_idx), len(features_dict) + 1]

    ## sparse matrix
    sparse_matrix = csr_matrix((elements, (row, col)), shape=(len(features_idx), len(features_dict)))

    ## pack results
    results = {}
    results['idx'] = location
    results['sparse_matrix'] = sparse_matrix
    results['elements'] = elements
    results['dim'] = dim

    return results


def tokenExtraction(window_size_list, data, mode):

    """
    given data, extract the corresponing feature
    :param window_size_list: [list] define the window size of the n-grams
    :param data: [list] input data needs to be in a form of (tweet, tagging, ner) tuple
    :param mode: define how to extract features from the tweet
    :return: [list] extracted n-gram features
    """

    ngram_all = []
    ##### line here can be (text, tagging, ner) tuple
    for line in data:
        curr_ngram = []
        ## find the location of the TARGET
        ##### replace the data with TARGET
        line2 = line.copy()
        line = line['text_TARGET'].split(" ")
        try:
            target_index = [idx for idx, i in enumerate(line) if i == "<TARGET>"][0]
        except:
            print(line)
            print("[ERROR] didn't find <TARGET>")
            # print((tweet, tagging, ner))
            raise
        ## extract features
        for window_size in window_size_list:
            start_index = max(0, target_index - window_size)
            end_index = min(target_index+window_size, len(line))
            if mode == "TARGET_two_sides":
                extracted_token = " ".join(line[start_index:end_index])
                curr_ngram.append(extracted_token)
            if mode == "TARGET_one_side":
                if line[0] != "<TARGET>":
                    extracted_token = " ".join(line[start_index:target_index+1])
                    curr_ngram.append(extracted_token)
                if line[-1] != "<TARGET>":
                    extracted_token = " ".join(line[target_index:end_index])
                    curr_ngram.append(extracted_token)
            if mode == "all":
                for idx in range(len(line)-window_size+1):
                    extracted_token = " ".join(line[idx:idx+window_size])
                    curr_ngram.append(extracted_token)

        if curr_ngram:
            ngram_all.append(curr_ngram)
        else:
            ngram_all.append('<UNK>')

    return ngram_all


def convertFeature2Idx(actual_features, train_feature_dict):

    """
    convert actual features into idx
    :param actual_features: real features extracted from tweets
    :param train_feature_dict: train_ngram_dict
    :return:
    """

    features_idx = []
    for line in actual_features:
        curr_feature = []
        for token in line:
            try:
                curr_feature.append(train_feature_dict[token])
            except:
                curr_feature.append(train_feature_dict['<UNK>'])
        features_idx.append(curr_feature)

    return features_idx


def buildTrainDict(train_ngram_all, verbose=False, set_threshold=False, threshold=1):

    """
    build up train ngram dictionary
    :param train_ngram_all: all extracted ngram features
    :param verbose:
    :param set_threshold: if we want to move ngrams with low frequency
    :param threshold: define low frequency
    :return: [dict] train_ngram_dict
    """

    ## collect all features
    train_ngram_all_flatten = [j for i in train_ngram_all for j in i]
    train_ngram_counter = Counter(train_ngram_all_flatten)
    train_ngram_counter = [(ngram, train_ngram_counter[ngram]) for ngram in train_ngram_counter.keys()]
    train_ngram_counter = sorted(train_ngram_counter, key=lambda x: x[1], reverse=True)
    if verbose:
        print("[I] total ngram", len(train_ngram_all_flatten), "unique ngram", len(set(train_ngram_all_flatten)))
        print("[I] the most frequent tokens: ", train_ngram_counter[0:20])
    # if you want to use n-grams from the whole tweets, then a threshold might be needed
    if set_threshold:
        print("[W] threshold " + str(threshold) + " is used for filtering the ngrams")
        find_threshold_index = min([idx for idx, i in enumerate(train_ngram_counter) if i[1] == threshold])
        train_ngram_counter = train_ngram_counter[0:find_threshold_index]
        # print(train_ngram_counter[-1], len(train_ngram_counter))

    ## build up the dictionary
    train_ngram_dict = {}
    for idx, i in enumerate(train_ngram_counter):
        train_ngram_dict[i[0]] = idx
    train_ngram_dict['<UNK>'] = idx + 1

    return train_ngram_counter, train_ngram_dict


def trainLRModel(train_all, train_label, window_size_list, ngram_extract_mode, flag, save_model=False):

    """
    given cyber threat data with severe / non-severe label, train a LR classifier
    :param train_all: training data
    :param train_label: training label
    :param window_size_list: n-gram window size
    :param ngram_extract_mode:
    :param flag:
    :param save_model:
    :return:
    """

    ## extract n-grams
    train_ngram_all = tokenExtraction(window_size_list, train_all, mode=ngram_extract_mode)

    ## build up training dictionary
    train_ngram_counter, train_ngram_dict = buildTrainDict(train_ngram_all, verbose=False,
                                                           set_threshold=True, threshold=1)

    train_features_idx = convertFeature2Idx(train_ngram_all, train_ngram_dict)

    train_features_no_dup = []
    for line in train_features_idx:
        train_features_no_dup.append(list(set(line)))

    train_idx_sparse = convertToSparseMatrix(train_features_no_dup, train_ngram_dict)

    ## train LR
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(train_idx_sparse['sparse_matrix'], train_label)

    ## print out some info
    print('[I] logistic regression training completed.')
    print('[I] training set dimension: ' + str(np.shape(train_idx_sparse['sparse_matrix'])))

    ## save model
    if save_model:
        with open('./trained_model/'+flag+'_lr_model.pkl', 'wb') as f:
            pickle.dump(lr, f)
        with open('./trained_model/'+flag+'_train_ngram_counter.json', 'w') as f:
            json.dump(train_ngram_counter, f)
        with open('./trained_model/'+flag+'_train_ngram_dict.json', 'w') as f:
            json.dump(train_ngram_dict, f)
        print("[I] all model files have been saved.")

    return lr, train_ngram_dict


def evalLRModel(window_size_list, val_all, train_ngram_dict, ngram_extract_mode, model):

    """
    cyber threat existence classifier
    :param window_size_list: define feature extraction window size
    :param val_all: data to be tested
    :param ngram_extract_mode: how the features are extracted
    :return:
    """

    ## val features preparation
    # extract ngram
    val_ngram_all = tokenExtraction(window_size_list, val_all, mode=ngram_extract_mode)
    # convert into idx
    val_features_idx = convertFeature2Idx(val_ngram_all, train_ngram_dict)
    # remove duplicates in features for each data point
    val_features_no_dup = []
    for line in val_features_idx:
        val_features_no_dup.append(list(set(line)))
    # convert to sparse matrix
    val_idx_sparse = convertToSparseMatrix(val_features_no_dup, train_ngram_dict)

    ## send val sparse matrix for classification
    val_prob = model.predict_proba(val_idx_sparse['sparse_matrix'])

    return val_prob