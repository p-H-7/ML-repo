import numpy as np


def prob(y):
    y_values = list(set(x for x in y))
    size = float(len(y))
    score = 0
    if size == 0:
        score = 1.0
    else:
        for y_val in y_values:
            count = 0
            for i in range(len(y)):
                if y[i] == y_val:
                    count += 1
            p = count/size
            score += p*p
    return (1 - score)*size


def gini_index(y_left, y_right):
    n_instances = float(len(y_left) + len(y_right))
    gini = (prob(y_left) + prob(y_right))/(n_instances)
    return gini


def findBestAttributeGini(X_train, y_train):
    best_feature, best_threshold, min_gini = None, None, np.inf
    for feature in X_train.columns:
        thresholds = X_train[feature].unique().tolist()
        thresholds.sort()
        thresholds = thresholds[1:]
        for t in thresholds:
            y_left_idx = X_train[feature] < t
            y_left = y_train[y_left_idx]
            y_right = y_train[~y_left_idx]
            gini_t = gini_index(y_left, y_right)
            if gini_t < min_gini:
                min_gini = gini_t
                best_threshold = t
                best_feature = feature
    return {'feature': best_feature, 'threshold': best_threshold}


def splitGini(X_train, y_train, depth, max_depth):

    if depth == max_depth or len(X_train) < 2:
        return {'prediction': np.mean(y_train)}

    attr = findBestAttributeGini(X_train, y_train)
    left_idx = X_train[attr['feature']] < attr['threshold']
    attr['left'] = splitGini(X_train[left_idx], y_train[left_idx],
                             depth + 1, max_depth)
    attr['right'] = splitGini(X_train[~left_idx], y_train[~left_idx],
                              depth + 1, max_depth)
    attr['_prediction'] = np.mean(y_train)
    return attr
