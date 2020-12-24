from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

import numpy as np
import pandas as pd
import time

# 加载训练数据, 选择尺度为400
def load_train_data(prep_type = 'maxmin'):
    '''
    timescale = 400
    :return:
    '''
    train1 = '/Data/zkx/piddata/train_data/train_csv/nonstiction_400.csv'
    #train2 = '/Data/zkx/piddata/train_data/train_csv/weakstiction_200.csv'
    train3 = '/Data/zkx/piddata/train_data/train_csv/strongstiction_400.csv'
    t1 = pd.read_csv(train1)
    #t2 = pd.read_csv(train2)
    t3 = pd.read_csv(train3)
    t1 = t1.sample(n=len(t3), replace=False, random_state=7, axis=0)
    data = pd.concat((t1, t3))
    data.reset_index(inplace = True, drop = True)
    data = shuffle(data)
    trainlabel = np.array(data['label'])
    # Convert label to 0,1
    trainlabelnew = []
    for i in range(len(trainlabel)):
        if trainlabel[i] == 'strongstiction':
            trainlabelnew.append(1)
        else:
            trainlabelnew.append(0)
    train = data.drop('label', axis=1)
    traindata = train.iloc[:, range(0, len(train.columns), 4)]
    traindata = np.array(traindata)
    # Data processing
    if prep_type == 'maxmin':
        #transformer = preprocessing.RobustScaler().fit(traindata)
        #traindata = transformer.transform(traindata)
        min_max_scaler = preprocessing.MinMaxScaler()
        traindatanew = min_max_scaler.fit_transform(traindata)
    elif prep_type == 'l2norm':
        #transformer = preprocessing.RobustScaler().fit(traindata)
        #traindata = transformer.transform(traindata)
        traindatanew = preprocessing.normalize(traindata, norm='l2')
    elif prep_type == 'none':
        traindatanew = traindata
    return traindatanew, trainlabelnew

def load_test_data(prep_type = 'maxmin'):
    # timescale = 100
    stictionpath = '/Data/zkx/piddata/test_data/csv/samespan/nonstiction_100.csv'
    nonstictionpath = '/Data/zkx/piddata/test_data/csv/samespan/stiction_100.csv'
    t1 = pd.read_csv(stictionpath, index_col=0)
    t2 = pd.read_csv(nonstictionpath, index_col=0)
    data = pd.concat((t1, t2))
    data.reset_index(inplace=True, drop=True)
    #data = shuffle(data)
    testlabel = np.array(data['label'])
    # Convert label to 0,1
    testlabelnew = []
    for i in range(len(testlabel)):
        if testlabel[i] == 'stiction':
            testlabelnew.append(1)
        else:
            testlabelnew.append(0)
    testdata = np.array(data.drop('label', axis=1))
    if prep_type == 'maxmin':
        #transformer = preprocessing.RobustScaler().fit(testdata)
        #testdata = transformer.transform(testdata)
        min_max_scaler = preprocessing.MinMaxScaler()
        testdatanew = min_max_scaler.fit_transform(testdata)
    elif prep_type == 'l2norm':
        #transformer = preprocessing.RobustScaler().fit(testdata)
        #testdata = transformer.transform(testdata)
        testdatanew = preprocessing.normalize(testdata, norm='l2')
    elif prep_type == 'none':
        testdatanew = testdata
    return testdatanew, testlabelnew

if __name__ == '__main__':

    starttime = time.time()
    print('---------------------------------------')
    print('Running Python file: LR.py')
    print('---------------------------------------')

    Train_X, Train_Y = load_train_data(prep_type='none')
    Test_X, Test_Y = load_test_data(prep_type='none')
    # Data Processing

    # Training model
    clf = LogisticRegression(penalty = 'none', solver = 'lbfgs').fit(Train_X, Train_Y)
    # Testing model
    pred_label = clf.predict(Test_X)
    pred_prob = clf.predict_proba(Test_X)
    print('Predict Label: \n', pred_label)
    print('Real Label: \n', Test_Y)

    # Metrics
    print('####################################')
    print('############# Metrics ##############')
    # 1. accuracy: how many samples are correct
    acc = accuracy_score(Test_Y, pred_label)
    print('accuracy classification score:', acc)
    # 2. classification_report: main classification metrics
    target_names = ['non-stiction', 'stiction']
    print('classification_report:')
    print(classification_report(Test_Y, pred_label, target_names=target_names))
    # 3. confusion_matrix
    tn, fp, fn, tp = confusion_matrix(Test_Y, pred_label).ravel()
    print('confusion_matrix:')
    print('TN:', tn, 'TP:', tp, 'FN:', fn, 'TP:', tp)
    endtime = time.time()
    print('---------------------------------------')
    print('Finish')
    print('Running Time: ', round(endtime - starttime, 2))
    print('---------------------------------------')