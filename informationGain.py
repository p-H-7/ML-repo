import numpy as np
import math


def Entropy(y):
    y_values = list(set(x for x in y))
    size = float(len(y))
    entropy = 0
    if size != 0:
        for y_val in y_values:
            count = 0
            for i in range(len(y)):
                if y[i] == y_val:
                    count += 1
            p = count/size
            entropy += p*math.log(p, 2)
    return entropy*(-1)


def AvgEntropyOfChildren(y_left, y_right):
    n_instances = float(len(y_left) + len(y_right))
    Avg = (Entropy(y_left)*(float(len(y_left))) +
           Entropy(y_right)*float(len(y_right)))/(n_instances)
    return Avg


def infoGain(entropyData, AvgEntropyChild):
    return (entropyData - AvgEntropyChild)


def findBestAttributeInfoGain(X_train, y_train):
    best_feature, best_threshold, maxInfoGain = None, None, -1
    for feature in X_train.columns:
        thresholds = X_train[feature].unique().tolist()
        thresholds.sort()
        thresholds = thresholds[1:]
        for t in thresholds:
            y_left_idx = X_train[feature] < t
            y_left = y_train[y_left_idx]
            y_right = y_train[~y_left_idx]
            entopyDad = Entropy(y_train)
            AvgEn = AvgEntropyOfChildren(y_left, y_right)
            gain_t = infoGain(entopyDad, AvgEn)
            if gain_t > maxInfoGain:
                maxInfoGain = gain_t
                best_threshold = t
                best_feature = feature
    return {'feature': best_feature, 'threshold': best_threshold}


def splitInfoGain(X_train, y_train, depth, max_depth):

    if depth == max_depth or len(X_train) < 2:
        return {'prediction': np.mean(y_train)}

    attr = findBestAttributeInfoGain(X_train, y_train)
    left_idx = X_train[attr['feature']] < attr['threshold']
    attr['left'] = splitInfoGain(X_train[left_idx], y_train[left_idx],
                                 depth + 1, max_depth)
    attr['right'] = splitInfoGain(X_train[~left_idx], y_train[~left_idx],
                                  depth + 1, max_depth)
    attr['_prediction'] = np.mean(y_train)
    return attr
