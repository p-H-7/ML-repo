'''
    Assignment - 1
    Group - 1
    Haasita Pinnepu - 19CS30021
    Swapnika Piriya - 19CS30035

    Implementation takes around 10 mins, please be patient. Though steps are printed to allow you to keep track that the program is running.
    The solution for each question is printed in the file "Solution.txt"
    Plots are created as .png
    The information Gain impurity measure is implemented in the file informationGain.py
    Similarly, the Gini Tree impurity measure is implemented in the file GiniTree.py

    Please Run the main.py file.
    The functions are pretty self-explanatory in terms of naming.
'''


import GiniTree
import informationGain
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


class DataClass:

    def __init__(self, line):
        line = line.strip().split(',')

        self.Pregnancies = int(line[0])
        self.Glucose = int(line[1])
        self.BloodPressure = int(line[2])
        self.SkinThickness = int(line[3])
        self.Insulin = int(line[4])
        self.BMI = float(line[5])
        self.DiabetesPedigreeFunction = float(line[6])
        self.Age = int(line[7])
        self.outcome = int(line[8])

    def get(self):
        return self.__dict__


def predict(sample, tree):
    prediction = None
    while prediction is None:
        feature, threshold = tree['feature'], tree['threshold']
        if sample[feature] < threshold:
            tree = tree['left']
        else:
            tree = tree['right']
        prediction = tree.get('prediction', None)
    return prediction


def evaluate(X, y, tree):
    y_preds = X.apply(predict, axis='columns', tree=tree.copy())
    mse = np.sum((y - y_preds) ** 2)
    mse /= X.shape[0]
    return mse


def getXYFromDataframe(df):
    X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    y = df[['outcome']].to_numpy().squeeze()
    return X, y


def buildTree(df, max_depth, type):
    idx1 = int(df.shape[0] * 0.6)
    idx2 = int(df.shape[0] * 0.8)
    bestTree = None
    bestTreeTrainDf = None
    bestTreeTestDf = None
    bestTreeValDf = None
    trainError, testError, valError = [], [], []
    bestError = np.inf

    for _ in range(10):
        newdf = df.copy()
        newdf = newdf.iloc[np.random.permutation(len(newdf))]
        traindf = newdf[:idx1]
        valdf = newdf[idx1:idx2]
        testdf = newdf[idx2:]
        X_train, y_train = getXYFromDataframe(traindf)
        X_test, y_test = getXYFromDataframe(testdf)
        X_val, y_val = getXYFromDataframe(valdf)
        if type == 'InfoGain':
            tree = informationGain.splitInfoGain(
                X_train, y_train, 0, max_depth)
        else:
            tree = GiniTree.splitGini(X_train, y_train, 0, max_depth)
        curTrainError = evaluate(X_train, y_train, tree)
        curTestError = evaluate(X_test, y_test, tree)
        curValError = evaluate(X_val, y_val, tree)
        trainError.append(curTrainError)
        testError.append(curTestError)
        valError.append(curValError)
        if (curTestError < bestError):
            bestError = curTestError
            bestTree = tree
            bestTreeTrainDf = traindf
            bestTreeTestDf = testdf
            bestTreeValDf = valdf
    return bestTree, testError, trainError, valError, max_depth, bestTreeTrainDf, bestTreeTestDf, bestTreeValDf


def canPrune(tree, X, y):

    if len(X) == 0:
        return True
    y_pred = tree['_prediction']
    mse = np.sum((y - y_pred) ** 2)
    mse /= len(X)
    if mse <= evaluate(X, y, tree):
        return True
    return False


def pruneTreeUtil(tree, X, y):

    if 'prediction' in tree.keys():
        return
    if len(X) == 0:
        tree['prediction'] = tree['_prediction']
        return
    indexes = X[tree['feature']] < tree['threshold']
    pruneTreeUtil(tree['left'], X[indexes], y[indexes])
    pruneTreeUtil(tree['right'], X[~indexes], y[~indexes])
    if canPrune(tree, X, y):
        tree['prediction'] = tree['_prediction']
        return


def printTree(tree, level=0, conditional='if '):
    '''
    Utility to print the decision tree as if-then-else statements
    '''
    if 'prediction' in tree.keys():
        conditional = conditional.replace('if ', '')
        ret = '\t' * level + conditional + \
            'Outcome = ' + str(tree['prediction']) + '\n'
        return ret
    ret = '\t' * level + conditional + tree['feature']
    ret += ' < ' + str(tree['threshold']) + '\n'
    ret += printTree(tree['left'], level + 1, 'then if ')
    ret += printTree(tree['right'], level + 1, 'else if ')
    return ret


def visualise(trainError, testError, xlabel, ylabel, title, saveAs):
    '''
    Utility to plot trainError and testError
    '''
    plt.switch_backend('Agg')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(trainError, label='Training Error')
    plt.plot(testError, label='Testing Error')
    plt.title(title)
    plt.legend()
    plt.savefig(saveAs)
    plt.close()


def analyzeTree(treeData, dS):
    dS.write('Analyzing passed tree:\n')
    _, testError, trainError, _valError, max_depth, _trainDf, _testDf, _valDf = treeData
    dS.write('\tmax_depth = {0}'.format(max_depth))
    for i in range(len(testError)):
        dS.write('\tVariation: {0} -> trainError = {1}, testError = {2}\n'.format(
            i, trainError[i], testError[i]))
    dS.write('\tmeanTrainError = {0}, meanTestError = {1}\n'.format(np.mean(trainError),
                                                                    np.mean(testError)))

    dS.write('\tTherefore, Accuracy = {0}\n\n'.format(
        (1-np.mean(testError))*100))

    visualise(trainError, testError,
              xlabel='Variations for same depth',
              ylabel='Mean Squared Error',
              title='Mean Squared Error of variations of same maximum depth',
              saveAs='Assignment1/tree.png')

    return np.mean(testError)


def analyzeModel(df, dS, type):
    trainError = [None]
    testError = [None]
    bestDepth = None
    bestError = np.inf
    dS.write('Analyzing the decision tree model:\n')
    for max_depth in range(1, 21):
        _, curTestError, curTrainError, _curValError, _max_depth, _trainDf, _testDf, _valDf = buildTree(
            df, max_depth, type)
        curTrainError = np.mean(curTrainError)
        curTestError = np.mean(curTestError)
        trainError.append(curTrainError)
        testError.append(curTestError)
        if (curTestError < bestError):
            bestError = curTestError
            bestDepth = max_depth
        dS.write('\tmax_depth = {0} -> trainError = {1}, testError = {2}\n'.format(max_depth,
                                                                                   curTrainError,
                                                                                   curTestError))
        print('analy = {0}'.format(max_depth))

    visualise(trainError, testError,
              xlabel='Maximum Depth',
              ylabel='Mean Squared Error',
              title='Mean Squared Error of the decision tree for different maximum depths',
              saveAs='Assignment1/analyze.png')
    dS.write('\tTherefore, bestDepth = {0}\n'.format(bestDepth))


if __name__ == '__main__':
    Data = open('Assignment1/diabetes.csv', 'r')
    records = [x.strip() for x in Data.readlines()][2:]
    records = [DataClass(x).get() for x in records]
    df = pd.DataFrame(records)

    sol = open('Assignment1/Solution.txt', 'w')
    treeTXT = open('Assignment1/tree.txt', 'w')
    prunedTree = open('Assignment1/prunedTree.txt', 'w')
    print('here0')

    max_depth = 10  # Assume Max depth of tree to be 10.
    sol.write('Solution to Q1 and Q2:\n')
    treeDataInfo = buildTree(df, max_depth, 'InfoGain')
    sol.write('Using Impurity Measure - Information Gain\n')
    meanErrorInfo = analyzeTree(treeDataInfo, sol)
    print('here1')

    treeDataGini = buildTree(df, max_depth, 'Gini')
    sol.write('Using Impurity Measure - Gini Index\n')
    meanErrorGini = analyzeTree(treeDataGini, sol)

    if meanErrorGini < meanErrorInfo:
        sol.write(
            'Comparing the Accuracies, We go forward with the Tree obtained from using Gini Index\n')
        tree, _testError, _trainError, _valError, _max_depth, traindf, testdf, valdf = treeDataGini
        type = 'Gini'
        print('here2')
    else:
        sol.write(
            'Comparing the Accuracies, We go forward with the Tree obtained from using Information Gain\n')
        tree, _testError, _trainError, _valError, _max_depth, traindf, testdf, valdf = treeDataInfo
        type = 'InfoGain'
        print('here3')

    sol.write('Solution to Q3:\n')
    analyzeModel(df, sol, type)
    print('here4')

    sol.write('\nSolution to Q4:\n')
    X_train, y_train = getXYFromDataframe(traindf)
    X_test, y_test = getXYFromDataframe(testdf)
    X_val, y_val = getXYFromDataframe(valdf)
    print('here5')
    treeTXT.write(printTree(tree))
    print('here6')
    sol.write('\tBefore pruning:\n')
    sol.write('\t\ttrainError = {0}, valError = {1}, testError = {2}\n\n'.format(evaluate(X_train, y_train, tree),
                                                                                 evaluate(
        X_val, y_val, tree),
        evaluate(X_test, y_test, tree)))
    print('here7')

    pruneTreeUtil(tree, X_val, y_val)
    sol.write('\tRefer to prunedTree.txt file\n\n')
    print('here8')
    prunedTree.write(printTree(tree))
    sol.write('\tAfter pruning:\n')
    sol.write('\t\ttrainError = {0}, valError = {1}, testError = {2}\n'.format(evaluate(X_train, y_train, tree),
                                                                               evaluate(
                                                                                   X_val, y_val, tree),
                                                                               evaluate(X_test, y_test, tree)))
    print('here9')

    sol.write('\nSolution to Q4:\n')
    sol.write('\tRefer to tree.txt file\n')

    Data.close()
    sol.close()
    treeTXT.close()
    prunedTree.close()
