
import argparse

from sklearn import metrics

from utils.io import *

from model.lr import *
from model.evaluation import *


if __name__ == "__main__":

    ## parse input
    parser = argparse.ArgumentParser()
    parser.add_argument('classifier_mode')
    parser.add_argument('eval_path')
    parser.add_argument('--window_size_list', '-w', nargs='+', type=int, default=[2, 3, 4])
    parser.add_argument('--ngram_extract_mode', '-e', default='all')
    options = parser.parse_args()

    ## existence classifier
    if options.classifier_mode == 'existence':

        ### load threat existence model
        try:
            with open('./trained_model/'+options.classifier_mode+'_lr_model.pkl', 'rb') as f:
                lr = pickle.load(f)
        except FileNotFoundError as err:
            print("[ERROR] model file doesn't exist", err)
            raise

        try:
            with open('./trained_model/'+options.classifier_mode+'_train_ngram_dict.json', 'r') as f:
                train_ngram_dict = json.load(f)
        except FileNotFoundError as err:
            print("[ERROR] train dict file doesn't exist", err)
            raise

        print('[I] loading complete.')

        ## read input
        exist_label2num = {'have_threat': 1, 'no_threat': 0}
        eval_all = readJSONFile(options.eval_path)

        ## evaluate model
        eval_prob = evalLRModel(options.window_size_list, eval_all,
                                train_ngram_dict, options.ngram_extract_mode, lr)

        ## append score
        for idx, each_line in enumerate(eval_all):
            each_line['existence_prob'] = eval_prob[idx][1]

        # ## evaluation
        # eval_label = [exist_label2num[i['existence_anno']] for i in eval_all]
        # eval_p, eval_r = calPR(eval_label, [i[1] for i in eval_prob])
        # print('auc', metrics.auc(eval_r, eval_p))

        ## write output
        writeJSONFile(eval_all, options.eval_path.replace('.json', '_with_score.json'), verbose=True)

    ## severity classifier
    if options.classifier_mode == 'severity':

        ## load threat existence model
        try:
            with open('./trained_model/'+options.classifier_mode+'_lr_model.pkl', 'rb') as f:
                lr = pickle.load(f)
        except FileNotFoundError as err:
            print("[ERROR] model file doesn't exist", err)
            raise

        try:
            with open('./trained_model/'+options.classifier_mode+'_train_ngram_dict.json', 'r') as f:
                train_ngram_dict = json.load(f)
        except FileNotFoundError as err:
            print("[ERROR] train dict file doesn't exist", err)
            raise

        print('[I] loading completed.')

        ## read input
        severity_label2num = {'severe': 1, 'not_severe': 0}
        eval_all = readJSONFile(options.eval_path)

        ## evaluate model
        eval_prob = evalLRModel(options.window_size_list, eval_all, train_ngram_dict,
                                options.ngram_extract_mode, lr)

        ## append score
        for idx, each_line in enumerate(eval_all):
            each_line['severity_prob'] = eval_prob[idx][1]

        ## evaluation
        # eval_label = [severity_label2num[i['severity_anno']] for i in eval_all]
        # eval_p, eval_r = calPR(eval_label, [i[1] for i in eval_prob])
        # print('auc', metrics.auc(eval_r, eval_p))

        ## write output
        writeJSONFile(eval_all, options.eval_path.replace('.json', '_with_score.json'), verbose=True)

        # ## print top ranked features
        # printTopFeatures(train_ngram_dict, lr)