import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import numpy as np
import pandas as pd
import time

def load_train_data(prep_type = 'maxmin', seed = 4):
    '''
    timescale = 400
    :return:
    '''
    train1 = '/Data/zkx/piddata/train_data/train_csv/nonstiction_400.csv'
    train3 = '/Data/zkx/piddata/train_data/train_csv/strongstiction_400.csv'
    t1 = pd.read_csv(train1)
    t3 = pd.read_csv(train3)
    t1 = t1.sample(n=len(t3)-30, replace=False, random_state=seed, axis=0)
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
    traindata = train.iloc[:, range(0, len(train.columns), 2)]
    traindata = np.array(traindata)
    # Data processing
    if prep_type == 'maxmin':
        min_max_scaler = preprocessing.MinMaxScaler()
        traindatanew = min_max_scaler.fit_transform(traindata)
    elif prep_type == 'l2norm':
        traindatanew = preprocessing.normalize(traindata, norm='l2')
    elif prep_type == 'none':
        traindatanew = traindata
    else:
        traindatanew = traindata
    return traindatanew, trainlabelnew

def load_test_data(prep_type = 'maxmin'):
    # timescale = 100
    stictionpath = '/Data/zkx/piddata/test_data/csv/samespan/nonstiction_200.csv'
    nonstictionpath = '/Data/zkx/piddata/test_data/csv/samespan/stiction_200.csv'
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
        min_max_scaler = preprocessing.MinMaxScaler()
        testdatanew = min_max_scaler.fit_transform(testdata)
    elif prep_type == 'l2norm':
        testdatanew = preprocessing.normalize(testdata, norm='l2')
    elif prep_type == 'none':
        testdatanew = testdata
    else:
        testdatanew = testdata
    return testdatanew, testlabelnew

def get_metrics(Test_Y, pred_label):
    # Metrics
    #print('############# Metrics ##############')
    #print('------------------------------------')
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
    print('TN:', tn, 'FP:', fp, 'FN:', fn, 'TP:', tp)

if __name__ == '__main__':

    starttime = time.time()
    print('---------------------------------------')
    print('Running Python file: Xgboost.py')
    print('---------------------------------------')

    Train_X, Train_Y = load_train_data(prep_type='l2norm')
    Test_X, Test_Y = load_test_data(prep_type='l2norm')
    # Xgboost Data Interface
    dtrain = xgb.DMatrix(Train_X, label=Train_Y)
    dtest = xgb.DMatrix(Test_X, label=Test_Y)
    # Model Parameters
    param = { 'booster': 'dart',
              'max_depth': 2,
              'eta': 1,
              'gamma': 6.4,
              'objective': 'binary:logistic'}
    # Training
    num_round = 4
    bst = xgb.train(param, dtrain, num_round)
    bst.save_model('/home/zhangkexin/code/work1/finalresult/Xgboost/xg.model')

    bst_reload = xgb.Booster(param)
    bst_reload.load_model('/home/zhangkexin/code/work1/finalresult/Xgboost/xg.model')
    # Testing and convert prob to label
    ypred = bst_reload.predict(dtest)
    pred_label = []
    for i in range(len(ypred)):
        if ypred[i] >= 0.5:
            pred_label.append(1)
        else:
            pred_label.append(0)

    get_metrics(Test_Y,pred_label)


    endtime = time.time()
    print('---------------------------------------')
    print('Finish')
    print('Running Time: ', round(endtime - starttime, 2))
    print('---------------------------------------')