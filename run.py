from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb
from model import LeNet5
from model import MTFL_train_test

from dataload.load_train_data import train_data_concat
from dataload.load_train_data import train_data_generate_image_lr
from dataload.load_train_data import train_data_generate_image_cnn
from dataload.load_train_data import train_data_generate_csv
from dataload.load_train_data import train_data_generate_fcnn
from dataload.load_train_data import train_data_generate_fcnn_lr

from dataload.load_test_data import test_data_generate_csv_fixedscales
from dataload.load_test_data import test_data_generate_image_fixedscales
from dataload.load_test_data import test_data_generate_image_mftcnn
from dataload.load_test_data import test_data_generate_image_fcnn
from dataload.load_test_data import test_data_prepare

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn import preprocessing

import numpy as np
import pandas as pd
import time
import pickle

def load_train_data(prep_type = 'l2norm'):
    '''
    timescale = 400
    :return:
    '''
    train1 = '/Data/zkx/piddata/train_data/train_csv/nonstiction_200.csv'
    train2 = '/Data/zkx/piddata/train_data/train_csv/strongstiction_200.csv'
    t1 = pd.read_csv(train1)
    t2 = pd.read_csv(train2)
    data = pd.concat((t1, t2))
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
    traindata = np.array(data.drop('label', axis=1))
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

def load_test_data(prep_type = 'l2norm'):
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
            testlabelnew.append(0)
        else:
            testlabelnew.append(1)
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

def load_MTCNN_data(reload = False):
    if reload:
        ##############################
        # ------- TRAIN DATA------------
        train_data_concat(norm_frac=0.9, valve_frac=0.7, weak_frac=0.5)
        '''
        /Data/zkx/piddata/train_data/matlab_data_concat/weakdata.csv
        /Data/zkx/piddata/train_data/matlab_data_concat/normaldata.csv
        /Data/zkx/piddata/train_data/matlab_data_concat/valvedata.csv
        '''
        ##############################
        # ------- Train -----------
        train_data_generate_csv(100, 200, 400, 600, sample=3)  # for LR/RF/SVM/Xgboost
        ''' 
        out: used for training LR, RF, SVM, Xgboost
        /Data/zkx/piddata/train_data/train_csv/nonstiction_*span.csv
        /Data/zkx/piddata/train_data/train_csv/strongstiction_*span.csv
        '''
        train_data_generate_image_cnn(200, 300, 400, 600, sample=3)  # for lenet5/mtfcnn
        '''
        out1: used for training CNN-MLR's CNN part
        /Data/zkx/piddata/train_data/train_image/jpg/nonstiction/normal_oppv*.jpg
        /Data/zkx/piddata/train_data/train_image/jpg/stiction/strong_oppv*.jpg
        out2: used for training CNN-MLR's MLR part
        /Data/zkx/piddata/train_data/train_image/jpg_mtfcnn/nonstiction/200/normal_oppv0200*.jpg
        /Data/zkx/piddata/train_data/train_image/jpg_mtfcnn/nonstiction/300/normal_oppv0300*.jpg
        /Data/zkx/piddata/train_data/train_image/jpg_mtfcnn/nonstiction/400/normal_oppv0400*.jpg
        /Data/zkx/piddata/train_data/train_image/jpg_mtfcnn/nonstiction/600/normal_oppv0600*.jpg
        /Data/zkx/piddata/train_data/train_image/jpg_mtfcnn/stiction/200/strong_oppv0200*.jpg
        /Data/zkx/piddata/train_data/train_image/jpg_mtfcnn/stiction/300/strong_oppv0300*.jpg
        /Data/zkx/piddata/train_data/train_image/jpg_mtfcnn/stiction/400/strong_oppv0400*.jpg
        /Data/zkx/piddata/train_data/train_image/jpg_mtfcnn/stiction/600/strong_oppv0600*.jpg
        '''
        train_data_generate_image_lr(50, 100, 150, 250, sample=3)  # for lenet5/mtfcnn
        '''
        out: used for valid CNN-MLR, get F1-score in validation datset
        /Data/zkx/piddata/train_data/train_image/jpg_mtfcnn_lr/nonstiction/*span
        /Data/zkx/piddata/train_data/train_image/jpg_mtfcnn_lr/stiction/*span
        '''
        train_data_generate_fcnn(200, 300, 400, 600, sample=3)  # Matrics for lenet5/mtfcnn
        '''
        out: used for training MatCNN-MLR
        /Data/zkx/piddata/train_data/train_image/jpg_matrics/nonstiction/*span
        /Data/zkx/piddata/train_data/train_image/jpg_matrics/stiction/*span
        '''
        train_data_generate_fcnn_lr(50, 100, 150, 250, sample=3)  # Matrics for lenet5/mtfcnn
        '''
        out: used for valid matrics, get F1-score in validation dataset
        /Data/zkx/piddata/train_data/train_image/jpg_matrics_lr/nonstiction/*span/
        /Data/zkx/piddata/train_data/train_image/jpg_matrics_lr/stiction/*span/
        '''
        ##############################
        # ------- TEST DATA------------
        test_data_prepare(30, seed=1)
        '''
        out: 
        /Data/zkx/piddata/test_data/csv/nonstictiondf.csv
        /Data/zkx/piddata/test_data/csv/stictiondf.csv
        '''
        test_data_generate_csv_fixedscales(50, 75, 100, 200, maxminflag=True)  # for LR/RF/SVM/Xgboost
        '''
        out: used for testing LR, RF, SVM, Xgboost
        /Data/zkx/piddata/test_data/csv/samespan/nonstiction_*span.csv
        /Data/zkx/piddata/test_data/csv/samespan/stiction_*span.csv
        '''
        test_data_generate_image_fixedscales()  # for lenet5
        ''' used for testing Single LeNet5, spans are same as .csv 
        /Data/zkx/piddata/test_data/image/samespan/nonstiction_*span/*loops.jpg.csv
        /Data/zkx/piddata/test_data/image/samespan/stiction_*span/*loops.jpg.csv
        '''
        test_data_generate_image_mftcnn(50, 75, 100, 200)  # for mftcnn fixed scales
        ''' used for testing CNN-MLR 
        /Data/zkx/piddata/test_data/image/test_image_mtfcnn/nonstiction/*loops/*loopname_*span.jpg
        /Data/zkx/piddata/test_data/image/test_image_mtfcnn/stiction/*loops/*loopname_*span.jpg
        '''
        test_data_generate_image_fcnn(50, 100, 150, 200)  # for mftcnn fixed scales
        ''' used for testing MatCNN-MLR 
        /Data/zkx/piddata/test_data/image/test_image_fcnn/stiction/*loops/*loopname_*span.npy
        /Data/zkx/piddata/test_data/image/test_image_fcnn/stiction/*loops/*loopname_*span.npy
        '''
        print('Re-Load Data Finish')

def get_metrics(Test_Y, pred_label):
    # Metrics
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
    print('------------------------------------')

if __name__ == '__main__':
    starttime = time.time()

    # Prepare Experimental Data
    load_MTCNN_data(reload=False) # for single LeNet-5, MTCNN-LR
    Train_X, Train_Y = load_train_data(prep_type='l2norm') # for LR, RF, SVM, Xgboost
    Test_X, Test_Y = load_test_data(prep_type='l2norm') # for LR, RF, SVM, Xgboost

    # --------- LR --------------
    # Training model
    clf = LogisticRegression(penalty='l2', solver='newton-cg').fit(Train_X, Train_Y)
    # Testing model
    pred_label = clf.predict(Test_X)
    # Get Metrics
    print('\n---- LR Experimental Results----')
    get_metrics(Test_Y, pred_label)

    # --------- RF --------------
    # Training model
    clf = RandomForestClassifier(n_estimators=200, max_depth=7).fit(Train_X, Train_Y)
    # Testing model
    pred_label = clf.predict(Test_X)
    # Get Metrics
    print('\n---- RF Experimental Results ----')
    get_metrics(Test_Y, pred_label)

    # --------- SVM -------------
    # Training model
    clf = svm.SVC(C=0.5, kernel='rbf', degree=4, gamma='scale', tol=1e-3, max_iter=-1).fit(Train_X, Train_Y)
    # Testing model
    pred_label = clf.predict(Test_X)
    # Get_metrics
    print('\n---- SVM Experimental Results ----')
    get_metrics(Test_Y, pred_label)

    # ------ Xgboost ------------
    # data transformation
    dtrain = xgb.DMatrix(Train_X, label=Train_Y)
    dtest = xgb.DMatrix(Test_X, label=Test_Y)
    # Model Parameters
    param = {'booster': 'gbtree',
             'max_depth': 3,
             'eta': 1,
             'gamma': 6.4,
             'objective': 'binary:logistic'}
    # Training
    num_round = 4
    bst = xgb.train(param, dtrain, num_round)
    # Testing and convert prob to label
    ypred = bst.predict(dtest)
    pred_label = []
    for i in range(len(ypred)):
        if ypred[i] >= 0.5:
            pred_label.append(1)
        else:
            pred_label.append(0)
    print('\n---- XgBoost Experimental Results ----')
    get_metrics(Test_Y, pred_label)

    # ------- LeNet5 -----------
    # Training model
    LeNet5.lenet5_train(epochs=8, train_batch=16)
    # Testing model
    Test_Y, pred_label = LeNet5.lenet5_test(testbatch=60)
    # Get Metrics
    print('---- LeNet5 Experimental Results----')
    get_metrics(Test_Y, pred_label)

    # ------- MTCNN-LR -----------
    print('---- MTFCNN Experimental Results----')
    # 测试第一个模型
    clf1 = MTFL_train_test.final_train_CNN_for_OPPV(epochs=8, train_batch_size=16)
    with open('/home/zhangkexin/code/work1/result/mtfcnn/sksave/clf1.pickle', 'wb') as f:
        pickle.dump(clf1, f)
    with open('/home/zhangkexin/code/work1/result/mtfcnn/sksave/clf1.pickle', 'rb') as f:
        clf1_reload = pickle.load(f)
    index1_0, index1_1 = MTFL_train_test.get_weight_OPPV(clf1_reload)
    pred_1, real_1, loopname1 = MTFL_train_test.final_test_CNN_for_OPPV(clf1_reload)
    print('Nonsticton index:', index1_0, 'Sticton index:', index1_1)

    # 测试第二个模型
    clf2 = MTFL_train_test.final_train_CNN_for_OPPV_dist(epochs=8, train_batch_size=16)
    with open('/home/zhangkexin/code/work1/result/mtfcnn/sksave/clf2.pickle', 'wb') as f:
        pickle.dump(clf2, f)
    with open('/home/zhangkexin/code/work1/result/mtfcnn/sksave/clf2.pickle', 'rb') as f:
        clf2_reload = pickle.load(f)
    index2_0, index2_1 = MTFL_train_test.get_weight_OPPV_dist(clf2_reload)
    pred_2, real_2, loopname2 = MTFL_train_test.final_test_CNN_for_OPPV_dist(clf2_reload)
    print('Nonsticton index:', index2_0, 'Sticton index:', index2_1)

    # 模型融合1
    feats, labels = MTFL_train_test.get_train_feats_fusion()
    clf = LogisticRegression(penalty='l2', solver='newton-cg').fit(feats, labels)
    with open('/home/zhangkexin/code/work1/result/mtfcnn/sksave/clf.pickle', 'wb') as f:
        pickle.dump(clf, f)
    with open('/home/zhangkexin/code/work1/result/mtfcnn/sksave/clf.pickle', 'rb') as f:
        clf_reload = pickle.load(f)
    f, l = MTFL_train_test.get_test_feats_fusion()
    final_pred = clf_reload.predict(f)
    print('-------------------------------------------------------')
    print('Fusion Model 1')
    get_metrics(l, final_pred)

    # 模型融合2
    print('-------------------------------------------------------')
    print('Fusion Model 2 ')
    if (index1_0 < index1_1) and (index2_0 > index2_1):
        pred = np.concatenate((pred_2[0: 30], pred_1[30:60]))
        real = real_1
        print('pred\n', pred)
        print('real\n', real)
    elif (index1_0 > index1_1) and (index2_0 < index2_1):
        pred = np.concatenate((pred_1[0: 30], pred_2[30:60]))
        real = real_1
        print('pred\n', pred)
        print('real\n', real)
    else:
        pred = final_pred
        real = real_1
        print('pred\n', pred)
        print('real\n', real)
    get_metrics(real, pred)

    endtime = time.time()
    print('---------------------------------------')
    print('Finish')
    print('Running Time: ', round(endtime - starttime, 2))
    print('---------------------------------------')
