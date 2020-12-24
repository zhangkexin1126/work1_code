import os
import re
import time
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

from PIL import Image

# Concat the simulation data: normal/weak/stiction
def train_data_concat(norm_frac = 1.0, valve_frac = 1.0, weak_frac = 1.0):
    seed = 1
    # weak
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_raw/weaks0p0.csv'
    data1 = pd.read_csv(loaddata, header=None)
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_raw/weaks0p1.csv'
    data2 = pd.read_csv(loaddata, header=None)
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_raw/weaks1p0.csv'
    data3 = pd.read_csv(loaddata, header=None)
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_raw/weaks1p1.csv'
    data4 = pd.read_csv(loaddata, header=None)
    # loaddata = r'/Data/zkx/piddata/matlabdata/csv/4/weaks1p3.csv'
    # data5 = pd.read_csv(loaddata, header=None)
    weakdata = pd.concat((data1, data2, data3, data4), axis=0)
    weakdata = weakdata.sample(frac=weak_frac, replace=False, random_state=seed, axis=0)
    print(len(weakdata))
    weakdata.to_csv(r'/Data/zkx/piddata/train_data/matlab_data_concat/weakdata.csv', index=False)


    # normal
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_raw/normals0p0.csv'
    data1 = pd.read_csv(loaddata, header=None)
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_raw/normals0p1.csv'
    data2 = pd.read_csv(loaddata, header=None)
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_raw/normals1p0.csv'
    data3 = pd.read_csv(loaddata, header=None)
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_raw/normals1p1.csv'
    data4 = pd.read_csv(loaddata, header=None)
    #loaddata = r'/Data/zkx/piddata/matlabdata/csv/4/normals1p3.csv'
    #data5 = pd.read_csv(loaddata, header=None)
    normaldata = pd.concat((data1, data2, data3, data4), axis=0)
    normaldata = normaldata.sample(frac=norm_frac, replace=False, random_state=seed, axis=0)
    print(len(normaldata))
    nonst = pd.concat((normaldata, weakdata), axis=0)
    normaldata.to_csv(r'/Data/zkx/piddata/train_data/matlab_data_concat/normaldata.csv', index = False)

    #valve
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_raw/valves0p0.csv'
    data1 = pd.read_csv(loaddata, header=None)
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_raw/valves0p1.csv'
    data2 = pd.read_csv(loaddata, header=None)
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_raw/valves1p0.csv'
    data3 = pd.read_csv(loaddata, header=None)
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_raw/valves1p1.csv'
    data4 = pd.read_csv(loaddata, header=None)
    #loaddata = r'/Data/zkx/piddata/matlabdata/csv/4/valves1p3.csv'
    #data5 = pd.read_csv(loaddata, header=None)
    valvedata = pd.concat((data1, data2, data3, data4), axis=0)
    valvedata = valvedata.sample(frac=valve_frac, replace=False, random_state=seed, axis=0)
    stictiondata = pd.concat((valvedata, weakdata), axis=0)
    print(len(valvedata))
    valvedata.to_csv(r'/Data/zkx/piddata/train_data/matlab_data_concat/valvedata.csv', index = False)
    # return normaldata, valvedata, weakdata

# Generate the train data --- csv
def train_data_generate_csv(*span, sample = 3):
    datalength = 1000
    datastart = 0
    dataspan = span
    samplespan = sample

    rootpath = '/Data/zkx/piddata/train_data/train_csv'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    shutil.rmtree(rootpath)
    os.makedirs(rootpath)

    # Normal data
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_concat/normaldata.csv'
    data = pd.read_csv(loaddata, header=0, index_col = False)
    op = data.iloc[:, 0:datalength]
    pv = data.iloc[:, datalength:datalength * 2]
    vv = data.iloc[:, datalength * 2:datalength * 3]
    sp = data.iloc[:, datalength * 3:datalength * 4]
    for span in dataspan:
        spa = str(span)
        savepath = '/Data/zkx/piddata/train_data/train_csv/' + 'nonstiction' + '_' + spa + '.csv'
        op = data.iloc[range(0, len(data), sample), datastart: datastart + span]
        pv = data.iloc[range(0, len(data), sample), datalength + datastart: datalength + datastart + span]
        train_data = pd.concat((op, pv), axis=1)
        # Add Label
        train_data['label'] = 'nonstiction'
        train_data.to_csv(savepath, index = False)

    # Strong data
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_concat/valvedata.csv'
    data = pd.read_csv(loaddata, header=0, index_col=False)
    op = data.iloc[:, 0:datalength]
    pv = data.iloc[:, datalength:datalength * 2]
    vv = data.iloc[:, datalength * 2:datalength * 3]
    sp = data.iloc[:, datalength * 3:datalength * 4]
    for span in dataspan:
        spa = str(span)
        savepath = '/Data/zkx/piddata/train_data/train_csv/' + 'strongstiction' + '_' + spa + '.csv'
        op = data.iloc[range(0, len(data), sample), datastart: datastart + span]
        pv = data.iloc[range(0, len(data), sample), datalength + datastart: datalength + datastart + span]
        train_data = pd.concat((op, pv), axis=1)
        # Add Label
        train_data['label'] = 'strongstiction'
        train_data.to_csv(savepath, index=False)

# Generate the train data --- image
def train_data_generate_image_cnn(*span, sample = 3):

    datalength = 1000
    plotstart = [0]
    plotspan = span
    samplespan = sample

    rootpath = '/Data/zkx/piddata/train_data/train_image/jpg'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    shutil.rmtree(rootpath)
    os.makedirs(rootpath)
    rootpath = '/Data/zkx/piddata/train_data/train_image/png'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    shutil.rmtree(rootpath)
    os.makedirs(rootpath)

    rootpath2 = '/Data/zkx/piddata/train_data/train_image/jpg_mtfcnn/nonstiction'
    if not os.path.exists(rootpath2):
        os.makedirs(rootpath2)
    shutil.rmtree(rootpath2)
    os.makedirs(rootpath2)

    rootpath3 = '/Data/zkx/piddata/train_data/train_image/jpg_mtfcnn/stiction'
    if not os.path.exists(rootpath3):
        os.makedirs(rootpath3)
    shutil.rmtree(rootpath3)
    os.makedirs(rootpath3)

    opvvflag = False
    # Normal
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_concat/normaldata.csv'
    data = pd.read_csv(loaddata, header=0, index_col = False)
    op = data.iloc[:, 0:datalength]
    pv = data.iloc[:, datalength:datalength * 2]
    vv = data.iloc[:, datalength * 2:datalength * 3]
    sp = data.iloc[:, datalength * 3:datalength * 4]
    savepath = r'/Data/zkx/piddata/train_data/train_image/png/nonstiction'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    shutil.rmtree(savepath)
    os.makedirs(savepath)
    for start in plotstart:
        sta = str(start)
        for span in plotspan:
            spa = str(span)
            pathcheck = os.path.join(rootpath2, spa)
            if not os.path.exists(pathcheck):
                os.makedirs(pathcheck)
            shutil.rmtree(pathcheck)
            os.makedirs(pathcheck)
            for i in range(0, len(data), samplespan):
                seq = str(i)
                opplot = op.iloc[i, start:start + span]
                pvplot = pv.iloc[i, start:start + span]
                vvplot = vv.iloc[i, start:start + span]
                savename = 'normal_oppv' + sta + spa + seq + '.png'
                savename2 = 'normal_oppv' + sta + spa + seq + '.jpg'
                datapath = os.path.join(savepath, savename)
                datapath2 = os.path.join(pathcheck, savename2)
                plt.figure(figsize=(2, 2), dpi=16)
                plt.plot(opplot, pvplot, 'k')
                #plt.xlim((0, 1))
                #plt.ylim((0, 1))
                plt.axis('off')
                plt.savefig(datapath)
                plt.savefig(datapath2)
                plt.close()
                if opvvflag:
                    savename = 'normal_opvv' + sta + spa + seq + '.png'
                    datapath = os.path.join(savepath, savename)
                    plt.figure(figsize=(2, 2), dpi=16)
                    plt.plot(opplot, vvplot, 'k')
                    plt.xlim((0, 1))
                    plt.ylim((0, 1))
                    plt.axis('off')
                    plt.savefig(datapath)
                    plt.close()

    # Valve
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_concat/valvedata.csv'
    data = pd.read_csv(loaddata, header=0, index_col = False)
    op = data.iloc[:, 0:datalength]
    pv = data.iloc[:, datalength:datalength * 2]
    vv = data.iloc[:, datalength * 2:datalength * 3]
    sp = data.iloc[:, datalength * 3:datalength * 4]
    savepath = r'/Data/zkx/piddata/train_data/train_image/png/stiction'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    shutil.rmtree(savepath)
    os.makedirs(savepath)
    for start in plotstart:
        sta = str(start)
        for span in plotspan:
            spa = str(span)
            pathcheck = os.path.join(rootpath3, spa)
            if not os.path.exists(pathcheck):
                os.makedirs(pathcheck)
            shutil.rmtree(pathcheck)
            os.makedirs(pathcheck)
            for i in range(0, len(data), samplespan):
                seq = str(i)
                opplot = op.iloc[i, start:start + span]
                pvplot = pv.iloc[i, start:start + span]
                vvplot = vv.iloc[i, start:start + span]
                savename = 'strong_oppv' + sta + spa + seq + '.png'
                savename2 = 'strong_oppv' + sta + spa + seq + '.jpg'
                datapath = os.path.join(savepath, savename)
                datapath2 = os.path.join(pathcheck, savename2)
                plt.figure(figsize=(2, 2), dpi=16)
                plt.plot(opplot, pvplot, 'k')
                #plt.xlim((0, 1))
                #plt.ylim((0, 1))
                plt.axis('off')
                plt.savefig(datapath)
                plt.savefig(datapath2)
                plt.close()
                if opvvflag:
                    savename = 'strong_opvv' + sta + spa + seq + '.png'
                    datapath = os.path.join(savepath, savename)
                    plt.figure(figsize=(2, 2), dpi=16)
                    plt.plot(opplot, vvplot, 'k')
                    plt.xlim((0, 1))
                    plt.ylim((0, 1))
                    plt.axis('off')
                    plt.savefig(datapath)

                    plt.close()

    # Convert to jpg
    normalpath = r'/Data/zkx/piddata/train_data/train_image/png/nonstiction'
    normalsavepath = r'/Data/zkx/piddata/train_data/train_image/jpg/nonstiction'
    if not os.path.exists(normalsavepath):
        os.makedirs(normalsavepath)
    shutil.rmtree(normalsavepath)
    os.makedirs(normalsavepath)
    matlabplotfile = os.listdir(normalpath)
    for i in range(len(matlabplotfile)):
        impath = os.path.join(normalpath, matlabplotfile[i])
        im = Image.open(impath)
        newfilename = matlabplotfile[i][0:-4] + '.jpg'
        savepath = os.path.join(normalsavepath, newfilename)
        rgb_im = im.convert('RGB')
        rgb_im.save(savepath)

    valvepath = r'/Data/zkx/piddata/train_data/train_image/png/stiction'
    valvesavepath = r'/Data/zkx/piddata/train_data/train_image/jpg/stiction'
    if not os.path.exists(valvesavepath):
        os.makedirs(valvesavepath)
    shutil.rmtree(valvesavepath)
    os.makedirs(valvesavepath)
    matlabplotfile = os.listdir(valvepath)
    for i in range(len(matlabplotfile)):
        impath = os.path.join(valvepath, matlabplotfile[i])
        im = Image.open(impath)
        newfilename = matlabplotfile[i][0:-4] + '.jpg'
        savepath = os.path.join(valvesavepath, newfilename)
        rgb_im = im.convert('RGB')
        rgb_im.save(savepath)

    print('number of normal train image: ', len(os.listdir(normalsavepath)))
    print('number of valve train image: ', len(os.listdir(valvesavepath)))

def train_data_generate_image_lr(*span, sample = 3):

    datalength = 1000
    plotstart = [0]
    plotspan = span
    samplespan = sample

    rootpath2 = '/Data/zkx/piddata/train_data/train_image/jpg_mtfcnn_lr/nonstiction'
    if not os.path.exists(rootpath2):
        os.makedirs(rootpath2)
    shutil.rmtree(rootpath2)
    os.makedirs(rootpath2)

    rootpath3 = '/Data/zkx/piddata/train_data/train_image/jpg_mtfcnn_lr/stiction'
    if not os.path.exists(rootpath3):
        os.makedirs(rootpath3)
    shutil.rmtree(rootpath3)
    os.makedirs(rootpath3)

    opvvflag = False
    # Normal
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_concat/normaldata.csv'
    data = pd.read_csv(loaddata, header=0, index_col = False)
    op = data.iloc[:, 0:datalength]
    pv = data.iloc[:, datalength:datalength * 2]
    vv = data.iloc[:, datalength * 2:datalength * 3]
    sp = data.iloc[:, datalength * 3:datalength * 4]
    for start in plotstart:
        sta = str(start)
        for span in plotspan:
            spa = str(span)
            pathcheck = os.path.join(rootpath2, spa)
            if not os.path.exists(pathcheck):
                os.makedirs(pathcheck)
            shutil.rmtree(pathcheck)
            os.makedirs(pathcheck)
            for i in range(0, len(data), samplespan):
                seq = str(i)
                opplot = op.iloc[i, start:start + span]
                pvplot = pv.iloc[i, start:start + span]
                vvplot = vv.iloc[i, start:start + span]
                savename2 = 'normal_oppv' + sta + spa + seq + '.jpg'
                datapath2 = os.path.join(pathcheck, savename2)
                plt.figure(figsize=(2, 2), dpi=16)
                plt.plot(opplot, pvplot, 'k')
                #plt.xlim((0, 1))
                #plt.ylim((0, 1))
                plt.axis('off')
                plt.savefig(datapath2)
                plt.close()


    # Valve
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_concat/valvedata.csv'
    data = pd.read_csv(loaddata, header=0, index_col = False)
    op = data.iloc[:, 0:datalength]
    pv = data.iloc[:, datalength:datalength * 2]
    vv = data.iloc[:, datalength * 2:datalength * 3]
    sp = data.iloc[:, datalength * 3:datalength * 4]
    for start in plotstart:
        sta = str(start)
        for span in plotspan:
            spa = str(span)
            pathcheck = os.path.join(rootpath3, spa)
            if not os.path.exists(pathcheck):
                os.makedirs(pathcheck)
            shutil.rmtree(pathcheck)
            os.makedirs(pathcheck)
            for i in range(0, len(data), samplespan):
                seq = str(i)
                opplot = op.iloc[i, start:start + span]
                pvplot = pv.iloc[i, start:start + span]
                vvplot = vv.iloc[i, start:start + span]
                savename2 = 'strong_oppv' + sta + spa + seq + '.jpg'
                datapath2 = os.path.join(pathcheck, savename2)
                plt.figure(figsize=(2, 2), dpi=16)
                plt.plot(opplot, pvplot, 'k')
                #plt.xlim((0, 1))
                #plt.ylim((0, 1))
                plt.axis('off')
                plt.savefig(datapath2)
                plt.close()

def train_data_generate_fcnn(*span, sample=3):
    datalength = 1000
    plotstart = [0]
    plotspan = span
    samplespan = sample
    mat_length = 50
    # Non-Stiction
    rootpath = '/Data/zkx/piddata/train_data/train_image/jpg_matrics/nonstiction'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    shutil.rmtree(rootpath)
    os.makedirs(rootpath)
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_concat/normaldata.csv'
    data = pd.read_csv(loaddata, header=0, index_col=False)
    op = data.iloc[:, 0:datalength]
    pv = data.iloc[:, datalength:datalength * 2]
    vv = data.iloc[:, datalength * 2:datalength * 3]
    sp = data.iloc[:, datalength * 3:datalength * 4]

    for start in plotstart:
        sta = str(start)
        for span in plotspan:
            spa = str(span)
            pathcheck = os.path.join(rootpath, spa)
            if not os.path.exists(pathcheck):
                os.makedirs(pathcheck)
            shutil.rmtree(pathcheck)
            os.makedirs(pathcheck)
            for i in range(0, len(data), samplespan):
                seq = str(i)
                mat = np.zeros((mat_length, mat_length), dtype=np.float)
                scale = np.int(span / mat_length)
                k = -1
                m = -1
                opplot = op.iloc[i, start:start + span].values
                pvplot = pv.iloc[i, start:start + span].values
                savename = 'normal_oppv' + sta + spa + seq
                datapath = os.path.join(pathcheck, savename)
                for i in range(0, len(opplot), scale):
                    k = k + 1
                    for j in range(0, len(pvplot), scale):
                        m = m + 1
                        opm = opplot[i: i + scale]
                        pvm = pvplot[j: j + scale]
                        mat[k][m] = np.sqrt(np.sum(np.square(opm - pvm)))
                    m = -1
                k = -1
                np.save(datapath, mat)

    # Non-Stiction
    rootpath = '/Data/zkx/piddata/train_data/train_image/jpg_matrics/stiction'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    shutil.rmtree(rootpath)
    os.makedirs(rootpath)
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_concat/valvedata.csv'
    data = pd.read_csv(loaddata, header=0, index_col=False)
    op = data.iloc[:, 0:datalength]
    pv = data.iloc[:, datalength:datalength * 2]
    vv = data.iloc[:, datalength * 2:datalength * 3]
    sp = data.iloc[:, datalength * 3:datalength * 4]
    for start in plotstart:
        sta = str(start)
        for span in plotspan:
            spa = str(span)
            pathcheck = os.path.join(rootpath, spa)
            if not os.path.exists(pathcheck):
                os.makedirs(pathcheck)
            shutil.rmtree(pathcheck)
            os.makedirs(pathcheck)
            for i in range(0, len(data), samplespan):
                seq = str(i)
                mat = np.zeros((mat_length, mat_length), dtype=np.float)
                scale = np.int(span / mat_length)
                k = -1
                m = -1
                opplot = op.iloc[i, start:start + span].values
                pvplot = pv.iloc[i, start:start + span].values
                savename = 'valve_oppv' + sta + spa + seq
                datapath = os.path.join(pathcheck, savename)
                for i in range(0, len(opplot), scale):
                    k = k + 1
                    for j in range(0, len(pvplot), scale):
                        m = m + 1
                        opm = opplot[i: i + scale]
                        pvm = pvplot[j: j + scale]
                        mat[k][m] = np.sqrt(np.sum(np.square(opm - pvm)))
                    m = -1
                k = -1
                np.save(datapath, mat)

def train_data_generate_fcnn_lr(*span, sample=3):
    datalength = 1000
    plotstart = [0]
    plotspan = span
    samplespan = sample
    mat_length = 50
    # Non-Stiction
    rootpath = '/Data/zkx/piddata/train_data/train_image/jpg_matrics_lr/nonstiction'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    shutil.rmtree(rootpath)
    os.makedirs(rootpath)
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_concat/normaldata.csv'
    data = pd.read_csv(loaddata, header=0, index_col=False)
    op = data.iloc[:, 0:datalength]
    pv = data.iloc[:, datalength:datalength * 2]
    vv = data.iloc[:, datalength * 2:datalength * 3]
    sp = data.iloc[:, datalength * 3:datalength * 4]

    for start in plotstart:
        sta = str(start)
        for span in plotspan:
            spa = str(span)
            pathcheck = os.path.join(rootpath, spa)
            if not os.path.exists(pathcheck):
                os.makedirs(pathcheck)
            shutil.rmtree(pathcheck)
            os.makedirs(pathcheck)
            for i in range(0, len(data), samplespan):
                seq = str(i)
                mat = np.zeros((mat_length, mat_length), dtype=np.float)
                scale = np.int(span / mat_length)
                k = -1
                m = -1
                opplot = op.iloc[i, start:start + span].values
                pvplot = pv.iloc[i, start:start + span].values
                savename = 'normal_oppv_lr' + sta + spa + seq
                datapath = os.path.join(pathcheck, savename)
                for i in range(0, len(opplot), scale):
                    k = k + 1
                    for j in range(0, len(pvplot), scale):
                        m = m + 1
                        opm = opplot[i: i + scale]
                        pvm = pvplot[j: j + scale]
                        mat[k][m] = np.sqrt(np.sum(np.square(opm - pvm)))
                    m = -1
                k = -1
                np.save(datapath, mat)

    # Non-Stiction
    rootpath = '/Data/zkx/piddata/train_data/train_image/jpg_matrics_lr/stiction'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    shutil.rmtree(rootpath)
    os.makedirs(rootpath)
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_concat/valvedata.csv'
    data = pd.read_csv(loaddata, header=0, index_col=False)
    op = data.iloc[:, 0:datalength]
    pv = data.iloc[:, datalength:datalength * 2]
    vv = data.iloc[:, datalength * 2:datalength * 3]
    sp = data.iloc[:, datalength * 3:datalength * 4]
    for start in plotstart:
        sta = str(start)
        for span in plotspan:
            spa = str(span)
            pathcheck = os.path.join(rootpath, spa)
            if not os.path.exists(pathcheck):
                os.makedirs(pathcheck)
            shutil.rmtree(pathcheck)
            os.makedirs(pathcheck)
            for i in range(0, len(data), samplespan):
                seq = str(i)
                mat = np.zeros((mat_length, mat_length), dtype=np.float)
                scale = np.int(span / mat_length)
                k = -1
                m = -1
                opplot = op.iloc[i, start:start + span].values
                pvplot = pv.iloc[i, start:start + span].values
                savename = 'valve_oppv-lr' + sta + spa + seq
                datapath = os.path.join(pathcheck, savename)
                for i in range(0, len(opplot), scale):
                    k = k + 1
                    for j in range(0, len(pvplot), scale):
                        m = m + 1
                        opm = opplot[i: i + scale]
                        pvm = pvplot[j: j + scale]
                        mat[k][m] = np.sqrt(np.sum(np.square(opm - pvm)))
                    m = -1
                k = -1
                np.save(datapath, mat)

####################################################
###################################################
# Generate the train data for MTFCNN ---np
# Convert .csv to .np
def train_data_generate_np_mftcnn():
    datalength = 1000
    #datastart = [start]
    #dataspan = [span1, span2, span3, span4, span5]
    #samplespan = sample

    rootpath = '/Data/zkx/piddata/train_data/train_np'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    shutil.rmtree(rootpath)
    os.makedirs(rootpath)

    # Normal data
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_concat/normaldata.csv'
    data = pd.read_csv(loaddata, header=0, index_col=False)
    op = data.iloc[:, 0:datalength].values
    pv = data.iloc[:, datalength:datalength * 2].values
    vv = data.iloc[:, datalength * 2:datalength * 3].values
    sp = data.iloc[:, datalength * 3:datalength * 4].values
    nonstiction_np = np.concatenate((op,pv),axis =1)
    savepath = os.path.join(rootpath, 'nonstiction')
    np.save(savepath, nonstiction_np)

    # Stictiond data
    loaddata = r'/Data/zkx/piddata/train_data/matlab_data_concat/valvedata.csv'
    data = pd.read_csv(loaddata, header=0, index_col=False)
    op = data.iloc[:, 0:datalength].values
    pv = data.iloc[:, datalength:datalength * 2].values
    vv = data.iloc[:, datalength * 2:datalength * 3].values
    sp = data.iloc[:, datalength * 3:datalength * 4].values
    stiction_np = np.concatenate((op, pv), axis=1)
    savepath = os.path.join(rootpath, 'stiction')
    np.save(savepath, stiction_np)



if __name__ == '__main__':

    starttime = time.time()
    print('---------------------------------------')
    print('Running Python file: load_train_data.py')
    print('---------------------------------------')

    #### Main_func
    train_data_concat(norm_frac=1.0, valve_frac=0.7, weak_frac=1.0)
    # Generate training images
    #train_data_generate_image(start=0, span1=200, span2=400, span3=600, span4=800, sample=3)
    # Generate training csv
    #train_data_generate_csv(start=0, span1=200, span2=400, span3=600, span4=800, sample=3)
    # Generate train np
    #train_data_generate_np_mftcnn()
    endtime = time.time()
    print('---------------------------------------')
    print('Finish')
    print('Running Time: ', round(endtime - starttime, 2))
    print('---------------------------------------')