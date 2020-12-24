import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import time
from torch.utils.data import DataLoader
from torch import optim
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from torchvision.datasets import DatasetFolder
import pickle
from torch.utils.data import random_split
from sklearn.metrics import f1_score

#########################################################
#########################################################
# --------- model1 --------------#
class Load_Dataset_for_OPPV(ImageFolder):
    def __init__(self, root, transform=None):
        super(Load_Dataset_for_OPPV, self).__init__(root, transform)

# 定义模型
class CNN_for_OPPV(nn.Module):
    def __init__(self, inchannel=1):
        super().__init__()
        self.inchannel = inchannel
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 25, 5)
        self.fc1 = nn.Linear(25 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 25 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        feat = x
        x = self.fc3(x)
        return x, feat

    def loss_func(self, pred, real):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, real)
        return {'loss': loss}


####### 验证阶段
# 根据验证集得到模型评分
def get_weight_OPPV(clf):
    # Load model
    PATH = '/home/zhangkexin/code/work1/finalresult/model_oppv.pth'
    finalnet = CNN_for_OPPV()
    finalnet.load_state_dict(torch.load(PATH))
    #
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 装载验证集
    corr = 0.8
    datapath = '/Data/zkx/piddata/train_data/train_image/jpg_mtfcnn_lr/nonstiction'
    ds = Load_Dataset_for_OPPV(datapath, transform)
    train_size = np.int(len(ds) * corr)
    modify_size = np.int(len(ds) - train_size)
    ds_train_non, ds_modify_non = random_split(ds, [train_size, modify_size])
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    featlist = []
    labellist = []
    with torch.no_grad():
        for data in dl:
            images, labels = data
            outputs = finalnet(images)[1]
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            for j in range(len(outputs)):
                featlist.append(outputs[j])
                labellist.append(labels[j])
    feat_in = np.array(featlist)
    label = labellist

    n_scale = np.int(len(ds.classes))
    length_scale = np.int(len(ds) / n_scale)
    # 重构feats
    feat1 = feat_in[0:length_scale, :]
    feat2 = feat_in[length_scale:length_scale * 2, :]
    feat3 = feat_in[length_scale * 2:length_scale * 3, :]
    feat4 = feat_in[length_scale * 3:length_scale * 4, :]
    nonstiction_feats = np.concatenate((feat1, feat2, feat3, feat4), axis=1)
    nonstiction_label = np.zeros(length_scale, dtype=int)

    datapath = '/Data/zkx/piddata/train_data/train_image/jpg_mtfcnn_lr/stiction'
    ds = Load_Dataset_for_OPPV(datapath, transform)
    train_size = np.int(len(ds) * corr)
    modify_size = np.int(len(ds) - train_size)
    ds_train_st, ds_modify_st = random_split(ds, [train_size, modify_size])
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    featlist = []
    labellist = []
    with torch.no_grad():
        for data in dl:
            images, labels = data
            outputs = finalnet(images)[1]
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            for j in range(len(outputs)):
                featlist.append(outputs[j])
                labellist.append(labels[j])

    feat_in = np.array(featlist)
    label = labellist
    n_scale = np.int(len(ds.classes))
    length_scale = np.int(len(ds) / n_scale)
    # 重构feats
    feat1 = feat_in[0:length_scale, :]
    feat2 = feat_in[length_scale:length_scale * 2, :]
    feat3 = feat_in[length_scale * 2:length_scale * 3, :]
    feat4 = feat_in[length_scale * 3:length_scale * 4, :]
    stiction_feats = np.concatenate((feat1, feat2, feat3, feat4), axis=1)

    stiction_label = np.ones(length_scale, dtype=int)
    N_valid = len(stiction_label)

    final_feats = np.concatenate((nonstiction_feats[0:N_valid, :], stiction_feats), axis=0)
    final_labels = np.concatenate((nonstiction_label[0:N_valid], stiction_label), axis=0)

    feat_pd = pd.DataFrame(final_feats)
    feat_pd['label'] = final_labels
    feat_pd = shuffle(feat_pd)
    feats = feat_pd.iloc[:, 0:-1].values
    labels = feat_pd['label'].values

    pred = clf.predict(feats)
    f1_stition = f1_score(labels, pred, pos_label=1)
    f1_nonstiction = f1_score(labels, pred, pos_label=0)
    return f1_nonstiction, f1_stition

# 装载第一个模型，并在多尺度测试集上进行运行，得到测试集的多尺度拼接特征，
# 默认为4个不同的尺度，如需调整，还需重写代码
def get_test_feats():
    # Load model
    PATH = '/home/zhangkexin/code/work1/finalresult/model_oppv.pth'
    finalnet = CNN_for_OPPV()
    finalnet.load_state_dict(torch.load(PATH))
    #
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    datapath = '/home/zhangkexin/code/work1/finalresult/test_image_mtfcnn/nonstiction'
    #datapath = '/Data/zkx/piddata/test_data/image/test_image_mtfcnn/nonstiction'
    ds = Load_Dataset_for_OPPV(datapath, transform)
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    featlist = []
    labellist = []
    with torch.no_grad():
        for data in dl:
            images, labels = data
            outputs = finalnet(images)[1]
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            # print(outputs.shape)
            for j in range(len(outputs)):
                featlist.append(outputs[j])
                labellist.append(labels[j])
    feat_in = np.array(featlist)
    label = labellist
    n_loop = np.int(len(ds.classes))
    n_scale = np.int(len(ds) / n_loop)
    length_scale = np.int(feat_in.shape[1])
    #print(n_loop, n_scale, length_scale)
    # 重构feats
    df = pd.DataFrame(np.zeros((n_loop, n_scale*length_scale), dtype=float))
    k=0
    for i in range(0, feat_in.shape[0], n_scale):
        feat1 = feat_in[i, :].reshape(1, -1)
        feat2 = feat_in[i + 1, :].reshape(1, -1)
        feat3 = feat_in[i + 2, :].reshape(1, -1)
        feat4 = feat_in[i + 3, :].reshape(1, -1)
        feat = np.concatenate((feat1, feat2, feat3, feat4), axis=1)
        df.iloc[k, :] = feat
        k = k+1
        df.index = ds.classes
    df['label'] = 0
    nonst = df

    #datapath = '/home/zhangkexin/code/work1/finalresult/mtfcnn/stiction'
    datapath = '/Data/zkx/piddata/test_data/image/test_image_mtfcnn/stiction'
    ds = Load_Dataset_for_OPPV(datapath, transform)
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    featlist = []
    labellist = []
    with torch.no_grad():
        for data in dl:
            images, labels = data
            outputs = finalnet(images)[1]
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            # print(outputs.shape)
            for j in range(len(outputs)):
                featlist.append(outputs[j])
                labellist.append(labels[j])
    feat_in = np.array(featlist)
    label = labellist
    n_loop = np.int(len(ds.classes))
    n_scale = np.int(len(ds) / n_loop)
    length_scale = np.int(feat_in.shape[1])
    #print(n_loop, n_scale, length_scale)
    # 重构feats
    df = pd.DataFrame(np.zeros((n_loop, n_scale * length_scale), dtype=float))
    k = 0
    for i in range(0, feat_in.shape[0], n_scale):
        feat1 = feat_in[i, :].reshape(1, -1)
        feat2 = feat_in[i + 1, :].reshape(1, -1)
        feat3 = feat_in[i + 2, :].reshape(1, -1)
        feat4 = feat_in[i + 3, :].reshape(1, -1)
        feat = np.concatenate((feat1, feat2, feat3, feat4), axis=1)
        df.iloc[k, :] = feat
        k = k + 1
        df.index = ds.classes
    df['label'] = 1
    st = df
    testdata = pd.concat((nonst,st), axis=0)
    loop = testdata.index
    feats = testdata.iloc[:, 0:-1].values
    label = testdata['label'].values

    return feats, label, loop
# 第一个模型完整测试过程，设定分类器为LR
def final_test_CNN_for_OPPV(clf):
    feats, real, loop = get_test_feats()
    pred = clf.predict(feats)
    print('-------------------------------------------------------')
    print('Model 1')
    # Metrics
    # 1. accuracy: how many samples are correct
    acc = accuracy_score(real, pred)
    print('accuracy classification score:', acc)
    # 2. classification_report: main classification metrics
    target_names = ['non-stiction', 'stiction']
    print('classification_report:')
    print(classification_report(real, pred, target_names=target_names))
    # 3. confusion_matrix
    tn, fp, fn, tp = confusion_matrix(real, pred).ravel()
    print('confusion_matrix:')
    print('TN:', tn, 'fP:', fp, 'FN:', fn, 'TP:', tp)
    return pred, real, loop
#########################################################
#########################################################

# --------- model2 --------------#
def myloader(path):
    item = np.load(path)
    return item

class Load_Dataset_for_OPPV_Dist(DatasetFolder):
    def __init__(self, root, loader, extensions, transform=None):
        super(Load_Dataset_for_OPPV_Dist, self).__init__(root, loader, extensions, transform)

class CNN_for_OPPV_Dist(nn.Module):
    def __init__(self, inchannel=1):
        super().__init__()

        self.inchannel = inchannel
        self.conv1 = nn.Conv2d(1, 8, 5, stride=2)
        self.conv2 = nn.Conv2d(8, 25, 5, stride=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(25 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 25 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        feat = x
        x = self.fc3(x)
        return x, feat

    def loss_func(self, pred, real):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, real)
        return {'loss': loss}

# 装载训练数据
def get_weight_OPPV_dist(clf):
    '''
    :param span:
    :return:
    '''
    # Load model
    PATH = '/home/zhangkexin/code/work1/finalresult/model_dist.pth'
    finalnet = CNN_for_OPPV_Dist()
    finalnet.load_state_dict(torch.load(PATH))
    transform = transforms.Compose(
        [transforms.ToTensor()])

    datapath = '/Data/zkx/piddata/train_data/train_image/jpg_matrics_lr/nonstiction'
    ds = Load_Dataset_for_OPPV_Dist(datapath, myloader, extensions='npy', transform=transform)
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    featlist = []
    labellist = []
    with torch.no_grad():
        for data in dl:
            images, labels = data
            outputs = finalnet(images.float())[1]
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            # print(outputs.shape)
            for j in range(len(outputs)):
                featlist.append(outputs[j])
                labellist.append(labels[j])
    feat_in = np.array(featlist)
    n_scale = np.int(len(ds.classes))
    length_scale = np.int(len(ds) / n_scale)
    # 重构feats
    feat1 = feat_in[0:length_scale, :]
    feat2 = feat_in[length_scale:length_scale * 2, :]
    feat3 = feat_in[length_scale * 2:length_scale * 3, :]
    feat4 = feat_in[length_scale * 3:length_scale * 4, :]
    nonstiction_feats = np.concatenate((feat1, feat2, feat3, feat4), axis=1)
    nonstiction_label = np.zeros(length_scale, dtype=int)

    datapath = '/Data/zkx/piddata/train_data/train_image/jpg_matrics_lr/stiction'
    ds = Load_Dataset_for_OPPV_Dist(datapath, myloader, extensions='npy', transform=transform)
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    featlist = []
    labellist = []
    with torch.no_grad():
        for data in dl:
            images, labels = data
            outputs = finalnet(images.float())[1]
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            for j in range(len(outputs)):
                featlist.append(outputs[j])
                labellist.append(labels[j])
    feat_in = np.array(featlist)
    n_scale = np.int(len(ds.classes))
    length_scale = np.int(len(ds) / n_scale)
    # 重构feats
    feat1 = feat_in[0:length_scale, :]
    feat2 = feat_in[length_scale:length_scale * 2, :]
    feat3 = feat_in[length_scale * 2:length_scale * 3, :]
    feat4 = feat_in[length_scale * 3:length_scale * 4, :]
    stiction_feats = np.concatenate((feat1, feat2, feat3, feat4), axis=1)
    stiction_label = np.ones(length_scale, dtype=int)
    N_valid = len(stiction_label)

    final_feats = np.concatenate((nonstiction_feats[0:N_valid, :], stiction_feats), axis=0)
    final_labels = np.concatenate((nonstiction_label[0:N_valid], stiction_label), axis=0)
    feat_pd = pd.DataFrame(final_feats)
    feat_pd['label'] = final_labels
    feat_pd = shuffle(feat_pd)
    feats = feat_pd.iloc[:, 0:-1].values
    labels = feat_pd['label'].values

    pred = clf.predict(feats)
    f1_stition = f1_score(labels, pred, pos_label=1)
    f1_nonstiction = f1_score(labels, pred, pos_label=0)

    return f1_nonstiction, f1_stition

def get_test_feats_dist():
    # Load model
    PATH = '/home/zhangkexin/code/work1/finalresult/model_dist.pth'
    finalnet = CNN_for_OPPV_Dist()
    finalnet.load_state_dict(torch.load(PATH))
    #
    transform = transforms.Compose(
        [transforms.ToTensor()])

    datapath = '/home/zhangkexin/code/work1/finalresult/test_image_fcnn/nonstiction'
    ds = Load_Dataset_for_OPPV_Dist(datapath, myloader, extensions='npy', transform=transform)
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    featlist = []
    labellist = []
    with torch.no_grad():
        for data in dl:
            images, labels = data
            outputs = finalnet(images.float())[1]
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            # print(outputs.shape)
            for j in range(len(outputs)):
                featlist.append(outputs[j])
                labellist.append(labels[j])
    feat_in = np.array(featlist)
    label = labellist
    n_loop = np.int(len(ds.classes))
    n_scale = np.int(len(ds) / n_loop)
    length_scale = np.int(feat_in.shape[1])
    #print(n_loop, n_scale, length_scale)
    # 重构feats
    df = pd.DataFrame(np.zeros((n_loop, n_scale*length_scale), dtype=float))
    k=0
    for i in range(0, feat_in.shape[0], n_scale):
        feat1 = feat_in[i, :].reshape(1, -1)
        feat2 = feat_in[i + 1, :].reshape(1, -1)
        feat3 = feat_in[i + 2, :].reshape(1, -1)
        feat4 = feat_in[i + 3, :].reshape(1, -1)
        feat = np.concatenate((feat1, feat2, feat3, feat4), axis=1)
        df.iloc[k, :] = feat
        k = k+1
        df.index = ds.classes
    df['label'] = 0
    nonst = df

    datapath = '/home/zhangkexin/code/work1/finalresult/test_image_fcnn/stiction'
    ds = Load_Dataset_for_OPPV_Dist(datapath, myloader, extensions='npy', transform=transform)
    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    featlist = []
    labellist = []
    with torch.no_grad():
        for data in dl:
            images, labels = data
            outputs = finalnet(images.float())[1]
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            # print(outputs.shape)
            for j in range(len(outputs)):
                featlist.append(outputs[j])
                labellist.append(labels[j])
    feat_in = np.array(featlist)
    label = labellist
    n_loop = np.int(len(ds.classes))
    n_scale = np.int(len(ds) / n_loop)
    length_scale = np.int(feat_in.shape[1])
    #print(n_loop, n_scale, length_scale)
    # 重构feats
    df = pd.DataFrame(np.zeros((n_loop, n_scale * length_scale), dtype=float))
    k = 0
    for i in range(0, feat_in.shape[0], n_scale):
        feat1 = feat_in[i, :].reshape(1, -1)
        feat2 = feat_in[i + 1, :].reshape(1, -1)
        feat3 = feat_in[i + 2, :].reshape(1, -1)
        feat4 = feat_in[i + 3, :].reshape(1, -1)
        feat = np.concatenate((feat1, feat2, feat3, feat4), axis=1)
        df.iloc[k, :] = feat
        k = k + 1
        df.index = ds.classes
    df['label'] = 1
    st = df

    testdata = pd.concat((nonst, st), axis=0)
    loop = testdata.index
    feats = testdata.iloc[:, 0:-1].values
    label = testdata['label'].values

    return feats, label, loop

def final_test_CNN_for_OPPV_dist(clf):
    feats, real, loop = get_test_feats_dist()
    pred = clf.predict(feats)
    # Metrics
    print('-------------------------------------------------------')
    print('Model 2')
    # 1. accuracy: how many samples are correct
    acc = accuracy_score(real, pred)
    print('accuracy classification score:', acc)
    # 2. classification_report: main classification metrics
    target_names = ['non-stiction', 'stiction']
    print('classification_report:')
    print(classification_report(real, pred, target_names=target_names))
    # 3. confusion_matrix
    tn, fp, fn, tp = confusion_matrix(real, pred).ravel()
    print('confusion_matrix:')
    print('TN:', tn, 'fP:', fp, 'FN:', fn, 'TP:', tp)

    return pred, real, loop

#########################################################
#########################################################
# ----------------------Fusion

def get_test_feats_fusion(w1=0.5, w2=0.5):
    # model_1
    feats_1, labels_1, loop = get_test_feats()
    # model_2
    feats_2, labels_2, loop = get_test_feats_dist()
    feats = np.concatenate((feats_1, feats_2), axis=1)
    labels = labels_1
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(feats)
    return data, labels

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
    print('------------------------------------')

if __name__ == '__main__':

    starttime = time.time()
    print('---------------------------------------')
    print('Running Python file: MTFL_train_test.py')
    print('---------------------------------------')

    # 测试第一个模型
    with open('/home/zhangkexin/code/work1/finalresult/sksave/clf1.pickle', 'rb') as f:
        clf1_reload = pickle.load(f)
    index1_0, index1_1 = get_weight_OPPV(clf1_reload)
    pred_1, real_1, loopname1 = final_test_CNN_for_OPPV(clf1_reload)
    print('Nonsticton index:', index1_0, 'Sticton index:', index1_1)

    # 测试第二个模型
    with open('/home/zhangkexin/code/work1/finalresult/mtfl_ts/3/clf2.pickle', 'rb') as f:
        clf2_reload = pickle.load(f)
    index2_0, index2_1 = get_weight_OPPV_dist(clf2_reload)
    pred_2, real_2, loopname2 = final_test_CNN_for_OPPV_dist(clf2_reload)
    print('Nonsticton index:', index2_0, 'Sticton index:', index2_1)

    # 模型融合1
    with open('/home/zhangkexin/code/work1/finalresult/sksave/clf.pickle', 'rb') as f:
        clf_reload = pickle.load(f)
    f, l = get_test_feats_fusion()
    final_pred = clf_reload.predict(f)
    print('-------------------------------------------------------')
    print('Fusion Model 1')
    get_metrics(l, final_pred)

    # 模型融合2
    print('-------------------------------------------------------')
    print('Fusion Model 2 ')
    if (index1_0 < index1_1) and  (index2_0 > index2_1):
        pred = np.concatenate((pred_2[0: 30], pred_1[30:60]))
        real = real_1
        print('pred\n', pred)
        print('real\n', real)
    elif (index1_0 > index1_1) and  (index2_0 < index2_1):
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

    # Other Info
    print('-------------------------------------------------------')
    print('Loops')
    print(loopname1)
    endtime = time.time()
    print('---------------------------------------')
    print('Finish')
    print('Running Time: ', round(endtime - starttime, 2))
    print('---------------------------------------')
