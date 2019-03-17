
### existence of threat classifier

import argparse

from utils.io import *
from model.lr import *

from sklearn import metrics
from model.evaluation import *


if __name__ == "__main__":

    ## parse input
    parser = argparse.ArgumentParser()
    parser.add_argument('classifier_mode')
    parser.add_argument('train_path')
    parser.add_argument('--window_size_list', '-w', nargs='+', type=int, default=[2, 3, 4])
    parser.add_argument('--ngram_extract_mode', '-e', default='all')
    options = parser.parse_args()

    ## existence classifier
    if options.classifier_mode == 'existence':

        exist_label2num = {'have_threat': 1, 'no_threat': 0}

        ## read input
        train_all = readJSONFile(options.train_path)
        train_label = [exist_label2num[i['existence_anno']] for i in train_all]

        ## train model
        exist_lr, exist_train_ngram_dict = trainLRModel(train_all, train_label,
                                                        options.window_size_list,
                                                        options.ngram_extract_mode,
                                                        flag='existence', save_model=True)

    ## severity classifier
    if options.classifier_mode == 'severity':

        severity_label2num = {'severe': 1, 'not_severe': 0}

        ## read input
        train_all = readJSONFile(options.train_path)
        train_label = [severity_label2num[i['severity_anno']] for i in train_all]

        ## train model
        lr, train_ngram_dict = trainLRModel(train_all, train_label,
                                            options.window_size_list,
                                            options.ngram_extract_mode,
                                            flag='severity', save_model=True)
