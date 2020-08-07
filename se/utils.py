import numpy as np
import csv
from sklearn import preprocessing

import random

categories = {
    '1': ['b', 'a'],
    '4': ['u', 'y', 'l', 't'],
    '5': ['g', 'p', 'gg'],
    '6': ['c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff'],
    '7': ['v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o'],
    '9': ['t', 'f'],
    '10': ['t', 'f'],
    '11': ['t', 'f'],
    '13': ['g', 'p', 's']
}

def preprocess_data(path):
    '''
    preprocessing reference from
    "https://stats.stackexchange.com/questions/82923/mixing-continuous-and-binary-data-with-linear-svm"

    NO 't' IN A4, SO ONLY 40 CATEGORIES IN DISCRETE FEATURE
    '''
    continuous_feature = []
    discrete_feature = []
    binary_feature = []
    label = []

    with open(path) as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            # continuous_feature.append([float(row[i]) for i in [1, 2, 7, 10, 13, 14]])
            continuous_feature.append([float(row[i]) if row[i] != '?' else np.nan for i in [1, 2, 7, 10, 13, 14]])
            discrete_feature.append([row[i] for i in [0, 3, 4, 5, 6, 12]])
            binary_feature.append([1. if row[i] == 't' else 0. for i in [8, 9, 11]])
            label.append([1. if row[-1] == '+' else 0.])

    # handling continuous data
    continuous_scaled = preprocessing.scale(continuous_feature)
    mu = np.nanmean(continuous_scaled, axis=0)
    for i in range(6):
        continuous_scaled[np.where(np.isnan(continuous_scaled[:, i]))] = mu[i]

    # handling categorical data
    enc = preprocessing.OneHotEncoder(handle_unknown='ignore').fit(discrete_feature)
    for i in range(len(enc.categories_)):
        if '?' in enc.categories_[i]:
            enc.categories_[i] = enc.categories_[i][1:]

    discrete_onehot = enc.transform(discrete_feature).toarray()
    binary_feature = np.array(binary_feature)

    feature_vector = np.concatenate([continuous_scaled, discrete_onehot, binary_feature], axis=1)
    label = np.array(label)

    indice = np.arange(len(label))
    random.shuffle(indice)

    feature_vector, label = feature_vector[indice], label[indice]

    print('shape of preprocessed data {}'.format(np.shape(feature_vector)))

    data_folds = []
    for i in range(10):
        data_folds.append((feature_vector[int(len(label) / 10 * i): int(len(label) / 10 * (i+1))],
                           label[int(len(label) / 10 * i): int(len(label) / 10 * (i+1))]))

    return data_folds

