
import numpy as np


def calPR(true_label, conf_score):

    """
    calculate precision / recall curve
    :param true_label: true labels
    :param conf_score: predictions scores
    :return: precision, recall values
    """

    combine = []
    for i in range(len(true_label)):
        combine.append((conf_score[i], true_label[i]))

    updated_prob_sorted = sorted(combine, key=lambda x: x[0], reverse=True)

    TP = 0
    FP = 0

    precision = []
    recall = []
    f1 = []

    if np.sum(true_label) == 0:
        print("[WARNING] no true label")
    else:
        for (prob, label) in updated_prob_sorted:
            if label == 1:
                TP += 1
            else:
                FP += 1

            pre_value = float(TP) / (TP + FP)
            rec_value = float(TP) / np.sum(true_label)
            if pre_value == 0 and rec_value == 0:
                f1_value = 0
            else:
                f1_value = 2 * (pre_value * rec_value) / (pre_value + rec_value)

            precision.append(pre_value)
            recall.append(rec_value)
            f1.append(f1_value)

    return precision, recall


def printTopFeatures(train_ngram_dict, lr):

    """
    evaluate top ranked features for logistic regression
    :param train_ngram_dict: trained n-gram dictionary
    :param lr: trained lr model
    :return: None
    """

    ## sort features by weights
    train_ngram_dict_reverse = {}
    for i in train_ngram_dict.items():
        train_ngram_dict_reverse[i[1]] = i[0]

    token_ranked_coef = sorted([(train_ngram_dict_reverse[i], lr.coef_[0][i]) \
                                for i in range(len(train_ngram_dict))], key=lambda x: x[1], reverse=True)
    ## print features
    for i in range(50):
        print(token_ranked_coef[i][0], "%.2f" % token_ranked_coef[i][1])

    return None
