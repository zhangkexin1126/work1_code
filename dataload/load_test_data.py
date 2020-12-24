import os
import re
import time
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn import preprocessing

# 整理TestData为df结构
def test_data_prepare(testnum = 25, seed = 1):
    '''
    :param testnum:
    :return: /Data/zkx/piddata/test_data/csv/nonstictiondf.csv
    :return: /Data/zkx/piddata/test_data/csv/stictiondf.csv
    '''
    # ------------------------------------------------------------------------------------------------------------------
    # 读取粘滞回路数据和回路名: stictiondata/stictionloop
    # ------------------------------------------------------------------------------------------------------------------
    filepath = r'/Data/zkx/piddata/test_data/stiction_loop_data'
    os.chdir(filepath)
    filelist = os.listdir(filepath)
    stictiondata = {}
    for i in range(len(filelist)):
        data_path = os.path.join(filepath, filelist[i])
        stictiondata[os.path.basename(data_path)[0:-4]] = pd.read_csv(data_path, header=None, skiprows=4,
                                                                      engine='python',
                                                                      names=[os.path.basename(data_path)[0:-4]])
    # print('There are {num} files in the path: {path}'.format(num=len(filelist), path=filepath))
    stictionloop = list(map(lambda x: x[0:-3], list(stictiondata.keys())))
    stictionloop = list(set(stictionloop))

    # 读取正常回路数据和回路名:normaldata/nomalloop
    filepath = r'/Data/zkx/piddata/test_data/normal_loop_data'
    os.chdir(filepath)
    filelist = os.listdir(filepath)
    normaldata = {}
    for i in range(len(filelist)):
        data_path = os.path.join(filepath, filelist[i])
        normaldata[os.path.basename(data_path)[0:-4]] = pd.read_csv(data_path, header=None, skiprows=4,
                                                                    engine='python',
                                                                    names=[os.path.basename(data_path)[0:-4]])
    # print('There are {num} files in the path: {path}'.format(num=len(filelist), path=filepath))
    normalloop = list(map(lambda x: x[0:-3], list(normaldata.keys())))
    normalloop = list(set(normalloop))

    # ------------------------------------------------------------------------------------------------------------------
    # 读取回路信息，包括回路样本数量和采样周期: loop1info
    # ------------------------------------------------------------------------------------------------------------------
    filepath = r'/Data/zkx/piddata/test_data/test_loop_info'
    filelist = os.listdir(filepath)
    loopinfo = pd.DataFrame(columns=['loopname', 'ts', 'samplenum'])
    pattern1 = re.compile('Ts: \d.+|Ts: \d+')
    pattern2 = re.compile(r'\d.+|\d+')
    pattern3 = re.compile(r'PV: \[\d+')
    pattern4 = re.compile(r'\d+')
    for i in range(len(filelist)):
        data_path = os.path.join(filepath, filelist[i])
        with open(data_path, 'r') as f:
            tstemp = f.readlines()
            numtemp = tstemp.copy()
            # 查询ts
            tstemp = list(map(lambda x: pattern1.search(x), tstemp))
            tstemp = list(filter(None, tstemp))[0]
            ts = pattern2.search(tstemp.group(0)).group(0)
            # 查询样本数量
            numtemp = list(map(lambda x: pattern3.search(x), numtemp))
            numtemp = list(filter(None, numtemp))[0]
            num = pattern4.search(numtemp.group(0)).group(0)
        loopinfo.loc[i, 'loopname'] = os.path.basename(data_path)[0:-4]
        loopinfo.loc[i, 'ts'] = ts
        loopinfo.loc[i, 'samplenum'] = num
    loopinfo.to_csv(r'/Data/zkx/piddata/test_data/loopinfo.csv')

    # ------------------------------------------------------------------------------------------------------------------
    # 将粘滞回路数据按照dict格式存放：stictiondata
    # ------------------------------------------------------------------------------------------------------------------
    datalist = list(stictiondata.keys())
    # stictionloop
    stictiondatadict = {}
    for loop in stictionloop:
        df = pd.DataFrame(columns=['op', 'pv', 'sp'])
        df.op = stictiondata[loop + '.OP'].values.flatten()
        df.pv = stictiondata[loop + '.PV'].values.flatten()
        df.sp = stictiondata[loop + '.SP'].values.flatten()
        stictiondatadict[loop] = df
    #print('Number of valveloop:', len(list(stictiondatadict.keys())))
    stictioninfo = loopinfo.loc[loopinfo['loopname'].isin(stictionloop)]

    # 将正常回路数据按照dict格式存放： normaldatadict
    datalist = list(normaldata.keys())
    # stictionloop
    normaldatadict = {}
    for loop in normalloop:
        df = pd.DataFrame(columns=['op', 'pv', 'sp'])
        df.op = normaldata[loop + '.OP'].values.flatten()
        df.pv = normaldata[loop + '.PV'].values.flatten()
        df.sp = normaldata[loop + '.SP'].values.flatten()
        normaldatadict[loop] = df
    #print('Number of normalloop:', len(list(normaldatadict.keys())))
    normalinfo = loopinfo.loc[loopinfo['loopname'].isin(normalloop)]

    # ------------------------------------------------------------------------------------------------------------------
    # 设置参数
    # ------------------------------------------------------------------------------------------------------------------
    random.seed(seed)
    # ------------------------------------------------------------------------------------------------------------------
    # 删选数据集 按照粘滞回路测试集的数量随机选择正常回路
    # ------------------------------------------------------------------------------------------------------------------
    stickloopnum = len(stictiondatadict) - testnum
    deleteloop = random.sample(stictiondatadict.keys(), stickloopnum)
    for name in deleteloop:
        stictiondatadict.pop(name)

    deleteloopnum = len(normaldatadict) - len(stictiondatadict)
    deleteloop = random.sample(normaldatadict.keys(), deleteloopnum)
    for name in deleteloop:
        normaldatadict.pop(name)

    #print('final test valve loop number: ', len(stictiondatadict))
    #print('final test normal loop number: ', len(normaldatadict))

    # ------------------------------------------------------------------------------------------------------------------
    # 每个回路选择200组样本存储, 考虑到最少的回路只有201个样本
    # ------------------------------------------------------------------------------------------------------------------
    savenum = 200
    stickdatadf = pd.DataFrame(np.zeros((len(stictiondatadict), 3 * savenum)))
    stickdatadf.index = list(stictiondatadict.keys())
    for k, v in stictiondatadict.items():
        stickdatadf.loc[k, 0:savenum - 1] = stictiondatadict[k].op[0:savenum].values
        stickdatadf.loc[k, savenum:savenum * 2 - 1] = stictiondatadict[k].pv[0:savenum].values
        stickdatadf.loc[k, savenum * 2:savenum * 3 - 1] = stictiondatadict[k].sp[0:savenum].values
    savepath = r'/Data/zkx/piddata/test_data/csv/stictiondf.csv'
    stickdatadf.to_csv(savepath)

    normaldatadf = pd.DataFrame(np.zeros((len(normaldatadict), 3 * savenum)))
    normaldatadf.index = list(normaldatadict.keys())
    for k, v in normaldatadict.items():
        normaldatadf.loc[k, 0:savenum - 1] = normaldatadict[k].op[0:savenum].values
        normaldatadf.loc[k, savenum:savenum * 2 - 1] = normaldatadict[k].pv[0:savenum].values
        normaldatadf.loc[k, savenum * 2:savenum * 3 - 1] = normaldatadict[k].sp[0:savenum].values
    savepath = r'/Data/zkx/piddata/test_data/csv/nonstictiondf.csv'
    normaldatadf.to_csv(savepath)

    return normaldatadf, stickdatadf, normaldatadict, stictiondatadict

# generate test data using fixed spans --- csv
def test_data_generate_csv_fixedscales(*span, maxminflag = True):
    '''
    :return: /Data/zkx/piddata/test_data/csv/samespan/...
    '''
    datalength = 200
    nonstictiondf, stictiondf, nonstictiondict, stictiondict = test_data_prepare(30)
    datastart = 0
    dataspan = span

    rootpath = '/Data/zkx/piddata/test_data/csv/samespan'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    shutil.rmtree(rootpath)
    os.makedirs(rootpath)

    max_min_scaler = preprocessing.MinMaxScaler()
    # Generate Nonstiction Data with fixed span
    data = nonstictiondf
    for span in dataspan:
        spa = str(span)
        savepath = '/Data/zkx/piddata/test_data/csv/samespan/' + 'nonstiction' + '_' + spa + '.csv'
        op1 = data.iloc[:, datastart: datastart + span]
        op = data.iloc[:, datastart: datastart + span].T
        looplist = op.columns.values
        opnew = max_min_scaler.fit_transform(op)
        opnew = pd.DataFrame(opnew, columns=looplist).T

        pv1 = data.iloc[:, datalength + datastart: datalength + datastart + span]
        pv = data.iloc[:, datalength + datastart: datalength + datastart + span].T
        looplist = pv.columns.values
        pvnew = max_min_scaler.fit_transform(pv)
        pvnew = pd.DataFrame(pvnew, columns=looplist).T

        if maxminflag:
            test_data = pd.concat((opnew, pvnew), axis=1)
        else:
            test_data = pd.concat((op1, pv1), axis=1)
        # Add Label
        test_data['label'] = 'nonstiction'
        test_data.to_csv(savepath)

    # Generate Stiction Data with fixed span
    data = stictiondf
    for span in dataspan:
        spa = str(span)
        savepath = '/Data/zkx/piddata/test_data/csv/samespan/' + 'stiction' + '_' + spa + '.csv'
        op1 = data.iloc[:, datastart: datastart + span]
        op = data.iloc[:, datastart: datastart + span].T
        looplist = op.columns.values
        opnew = max_min_scaler.fit_transform(op)
        opnew = pd.DataFrame(opnew, columns=looplist).T

        pv1 = data.iloc[:, datalength + datastart: datalength + datastart + span]
        pv = data.iloc[:, datalength + datastart: datalength + datastart + span].T
        looplist = pv.columns.values
        pvnew = max_min_scaler.fit_transform(pv)
        pvnew = pd.DataFrame(pvnew, columns=looplist).T
        if maxminflag:
            test_data = pd.concat((opnew, pvnew), axis=1)
        else:
            test_data = pd.concat((op1, pv1), axis=1)
        # Add Label
        test_data['label'] = 'stiction'
        test_data.to_csv(savepath)

def test_data_generate_image_fixedscales():
    # load_data
    '''
    :return: /Data/zkx/piddata/test_data/image/samespan
    '''
    rootpath = '/Data/zkx/piddata/test_data/image/samespan'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    shutil.rmtree(rootpath)
    os.makedirs(rootpath)
    csvpath = '/Data/zkx/piddata/test_data/csv/samespan'
    files = os.listdir(csvpath)
    for file in files:
        # 创建图像存储根目录
        filename = file[:-4]
        imagesavepath = '/Data/zkx/piddata/test_data/image/samespan/' + filename
        if not os.path.exists(imagesavepath):
            os.makedirs(imagesavepath)
        shutil.rmtree(imagesavepath)
        os.makedirs(imagesavepath)

        # 读取数据，准备绘制图像
        dataloadpath = csvpath + '/' + file
        data = pd.read_csv(dataloadpath, header=0, index_col=0)
        looplist = data.index
        # 绘制每个回路的图像数据
        for loop in looplist:
            dataspan = np.int((len(data.columns) - 1)/2)
            loopdata = data.loc[loop].values
            op = loopdata[0: dataspan]
            pv = loopdata[dataspan: dataspan*2]
            label = loopdata[-1]
            spa = str(dataspan)
            # 设置图像存储路径
            imagename = loop + '_' + spa + '.jpg'
            finalsavepath = os.path.join(imagesavepath, imagename)
            # 生成图像
            plt.figure(figsize=(2, 2), dpi=16)
            plt.plot(op, pv, 'k')
            #plt.xlim((0, 1))
            #plt.ylim((0, 1))
            plt.axis('off')
            plt.savefig(finalsavepath)
            plt.close()

    print('---------------------------------')
    print('Finish Generating Image with Fixed Timescale')

def test_data_generate_image_mftcnn(*span):

    rootpath = '/Data/zkx/piddata/test_data/image/test_image_mtfcnn/stiction'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    shutil.rmtree(rootpath)
    os.makedirs(rootpath)

    datalength = 200
    # Stiction Images
    loaddata = r'/Data/zkx/piddata/test_data/csv/samespan/stiction_200.csv'
    data = pd.read_csv(loaddata, header=0, index_col=0)
    looplist = data.index
    for loop in looplist:
        save = os.path.join(rootpath, loop)
        if not os.path.exists(save):
            os.makedirs(save)
        shutil.rmtree(save)
        os.makedirs(save)
        loopdata = data.loc[loop].values
        for scale in span:
            sca_str = str(scale)
            sca_int = int(scale)
            savename = loop + '_' + sca_str + '.jpg'
            savepath = os.path.join(save, savename)

            op = loopdata[0: sca_int]
            pv = loopdata[datalength: datalength + sca_int]
            plt.figure(figsize=(2, 2), dpi=16)
            plt.plot(op, pv, 'k')
            #plt.xlim((0, 1))
            #plt.ylim((0, 1))
            plt.axis('off')
            plt.savefig(savepath)
            plt.close()

    rootpath = '/Data/zkx/piddata/test_data/image/test_image_mtfcnn/nonstiction'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    shutil.rmtree(rootpath)
    os.makedirs(rootpath)
    # Non-Stiction Images
    loaddata = r'/Data/zkx/piddata/test_data/csv/samespan/nonstiction_200.csv'
    data = pd.read_csv(loaddata, header=0, index_col=0)
    looplist = data.index
    for loop in looplist:
        save = os.path.join(rootpath, loop)
        if not os.path.exists(save):
            os.makedirs(save)
        shutil.rmtree(save)
        os.makedirs(save)
        loopdata = data.loc[loop].values
        for scale in span:
            sca_str = str(scale)
            sca_int = int(scale)
            savename = loop + '_' + sca_str + '.jpg'
            savepath = os.path.join(save, savename)
            op = loopdata[0: sca_int]
            pv = loopdata[datalength: datalength + sca_int]
            plt.figure(figsize=(2, 2), dpi=16)
            plt.plot(op, pv, 'k')
            #plt.xlim((0, 1))
            #plt.ylim((0, 1))
            plt.axis('off')
            plt.savefig(savepath)
            plt.close()

def test_data_generate_image_fcnn(*span):
    datalength = 200
    mat_length = 50

    # Stiction Images
    rootpath = '/Data/zkx/piddata/test_data/image/test_image_fcnn/stiction'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    shutil.rmtree(rootpath)
    os.makedirs(rootpath)

    loaddata = r'/Data/zkx/piddata/test_data/csv/samespan/stiction_200.csv'
    data = pd.read_csv(loaddata, header=0, index_col=0)
    looplist = data.index
    for loop in looplist:
        save = os.path.join(rootpath, loop)
        if not os.path.exists(save):
            os.makedirs(save)
        shutil.rmtree(save)
        os.makedirs(save)
        loopdata = data.loc[loop].values
        for scale in span:
            for scale in span:
                sca_str = str(scale)
                sca_int = int(scale)
                savename = loop + '_' + sca_str
                savepath = os.path.join(save, savename)
                mat = np.zeros((mat_length, mat_length), dtype=np.float)
                n_scale = np.int(scale / mat_length)
                k = -1
                m = -1
                op = loopdata[0: sca_int]
                pv = loopdata[datalength: datalength + sca_int]
                for i in range(0, len(op), n_scale):
                    k = k + 1
                    for j in range(0, len(pv), n_scale):
                        m = m + 1
                        opm = op[i: i + n_scale]
                        pvm = pv[j: j + n_scale]
                        mat[k][m] = np.sqrt(np.sum(np.square(opm - pvm)))
                    m = -1
                k = -1
                np.save(savepath, mat)

    # NonStiction Images
    rootpath = '/Data/zkx/piddata/test_data/image/test_image_fcnn/nonstiction'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    shutil.rmtree(rootpath)
    os.makedirs(rootpath)

    loaddata = r'/Data/zkx/piddata/test_data/csv/samespan/nonstiction_200.csv'
    data = pd.read_csv(loaddata, header=0, index_col=0)
    looplist = data.index
    for loop in looplist:
        save = os.path.join(rootpath, loop)
        if not os.path.exists(save):
            os.makedirs(save)
        shutil.rmtree(save)
        os.makedirs(save)
        loopdata = data.loc[loop].values
        for scale in span:
            for scale in span:
                sca_str = str(scale)
                sca_int = int(scale)
                savename = loop + '_' + sca_str
                savepath = os.path.join(save, savename)
                mat = np.zeros((mat_length, mat_length), dtype=np.float)
                n_scale = np.int(scale / mat_length)
                k = -1
                m = -1
                op = loopdata[0: sca_int]
                pv = loopdata[datalength: datalength + sca_int]
                for i in range(0, len(op), n_scale):
                    k = k + 1
                    for j in range(0, len(pv), n_scale):
                        m = m + 1
                        opm = op[i: i + n_scale]
                        pvm = pv[j: j + n_scale]
                        mat[k][m] = np.sqrt(np.sum(np.square(opm - pvm)))
                    m = -1
                k = -1
                np.save(savepath, mat)

# generate test data using unfiaxed spans --- csv
def test_data_generate_csv_unfixedscales():
    # Load timescale for stictiondata
    rootpath = '/Data/zkx/piddata/test_data/csv/differspan'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    shutil.rmtree(rootpath)
    os.makedirs(rootpath)
    filepath = '/Data/zkx/piddata/test_data/stiction_timescale.txt'
    name = ['loop_num', 'loop_name', 'span1', 'span2', 'span3', 'span4', 'span5', 'span6', 'span7', 'span8', 'span9',
            'span10']
    ts = pd.read_csv(filepath, header=None, engine='python', sep='\t', names=name)
    ts.dropna(inplace=True)
    ts.reset_index(inplace=True, drop=True)
    ts.drop('loop_num', axis=1, inplace=True)
    ts.set_index('loop_name', inplace=True)
    # name of span
    spanlist = ts.columns.values
    # load data
    nonstictiondf, stictiondf, nonstictiondict, stictiondict = test_data_prepare(30)
    # Generate Stiction Data with unfixed span
    datadict = stictiondict
    max_min_scaler = preprocessing.MinMaxScaler()
    for span in spanlist:
        spa = span
        spanrootpath = os.path.join(rootpath,span)
        if not os.path.exists(spanrootpath):
            os.makedirs(spanrootpath)
        shutil.rmtree(spanrootpath)
        os.makedirs(spanrootpath)
        for k,v in datadict.items():
            #print(k,span)
            loopspan = ts.loc[[k], [span]].values[0][0]
            #print(loopspan)
            start, end = loopspan.split('-', 1)
            opnew = v.op[np.int(start):np.int(end)].values.reshape(-1,1)
            opnew = max_min_scaler.fit_transform(opnew)
            pvnew = v.pv[np.int(start):np.int(end)].values.reshape(-1,1)
            pvnew = max_min_scaler.fit_transform(pvnew)
            testdata = np.append(opnew, pvnew).reshape((1, -1))
            testdatadf = pd.DataFrame(testdata, index=[k])
            testdatadf['label'] = 'stiction'
            savepath = '/Data/zkx/piddata/test_data/csv/differspan/' + span + '/' + k + '.csv'
            testdatadf.to_csv(savepath)

# generate test data using oscillation timescales
def test_data_generate_image_unfixedscales():
    # load_data
    rootpath = '/Data/zkx/piddata/test_data/image/differspan'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    shutil.rmtree(rootpath)
    os.makedirs(rootpath)
    csvrootpath = '/Data/zkx/piddata/test_data/csv/differspan'
    csvlist = [] # 所有.csv数据文件的路径
    scalelist = [] # 所有子目录
    for root, dirs, files in os.walk(csvrootpath):
        for name in dirs:
            scalelist.append(os.path.join(root, name))
        for name in files:
            # print(os.path.join(root, name))
            csvlist.append(os.path.join(root, name))
    num = np.int(len(scalelist))
    x = list(np.array(csvlist).reshape(num, -1))
    for i, scaledir in enumerate(scalelist):
        # 创建路径
        scale = scaledir.split('/')[-1]
        imagesavepath = '/Data/zkx/piddata/test_data/image/differspan/' + scale
        if not os.path.exists(imagesavepath):
            os.makedirs(imagesavepath)
        shutil.rmtree(imagesavepath)
        os.makedirs(imagesavepath)
        # 读取数据
        datapath = x[i]
        for csvpath in datapath:
            data = pd.read_csv(csvpath, index_col = 0)
            datalength = np.int((len(data.columns) - 1)/2)
            loopdata = data.iloc[0,:].values
            op = loopdata[0: datalength]
            pv = loopdata[datalength: datalength * 2]
            label = loopdata[-1]
            # 设置图像存储路径
            loopname = data.index.values[0]
            imagename = loopname + '.jpg'
            finalsavepath = os.path.join(imagesavepath, imagename)
            # 生成图像
            plt.figure(figsize=(2, 2), dpi=16)
            plt.plot(op, pv, 'k')
            plt.xlim((0, 1))
            plt.ylim((0, 1))
            plt.axis('off')
            plt.savefig(finalsavepath)
            plt.close()

# generate test data for MTFCNN using fixed timescales -- np
# Convert .csv to .np
def test_data_generate_np_mftcnn():
    datalength = 200
    #datastart = [start]
    #dataspan = [span1, span2, span3, span4, span5]
    #samplespan = sample

    rootpath = '/Data/zkx/piddata/test_data/np'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    shutil.rmtree(rootpath)
    os.makedirs(rootpath)

    # Normal data
    loaddata = r'/Data/zkx/piddata/test_data/csv/samespan/nonstiction_200.csv'
    data = pd.read_csv(loaddata, header=0, index_col=0)
    op = data.iloc[:, 0:datalength].values
    pv = data.iloc[:, datalength:datalength * 2].values
    #label = data['label']
    nonstiction_np = np.concatenate((op,pv), axis =1)
    savepath = os.path.join(rootpath, 'nonstiction')
    np.save(savepath, nonstiction_np)

    # Stictiond data
    loaddata = r'/Data/zkx/piddata/test_data/csv/samespan/stiction_200.csv'
    data = pd.read_csv(loaddata, header=0, index_col=0)
    op = data.iloc[:, 0:datalength].values
    pv = data.iloc[:, datalength:datalength * 2].values
    # label = data['label']
    stiction_np = np.concatenate((op, pv), axis=1)
    savepath = os.path.join(rootpath, 'stiction')
    np.save(savepath, stiction_np)


if __name__ == '__main__':

    starttime = time.time()
    print('---------------------------------------')
    print('Running Python file: load_test_data.py')
    print('---------------------------------------')

    #### Main_func
    test_data_prepare(testnum = 30)
    ## CSV test data
    test_data_generate_csv_fixedscales(30, 70, 100, 200, maxminflag=True)
    #test_data_generate_csv_unfixedscales()
    ## Image test data
    test_data_generate_image_fixedscales()
    #test_data_generate_image_unfixedscales()
    test_data_generate_image_mftcnn(80, 120, 160, 200)

    endtime = time.time()
    print('---------------------------------------')
    print('Finish')
    print('Running Time: ', round(endtime - starttime, 2))
    print('---------------------------------------')