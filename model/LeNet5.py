import torch
import torchvision
from torch.utils.data import ConcatDataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import os
import shutil
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch import optim

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class lenet5(nn.Module):

    def __init__(self):
        super(lenet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def loss_func(self, pred_y, real_y):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred_y, real_y)

        return {'loss': loss}

class LoadImage_LeNet5(ImageFolder):
    def __init__(self, root, transform=None):
        super(LoadImage_LeNet5, self).__init__(root, transform)

def load_pid_train_data():
    datapath = '/Data/zkx/piddata/train_data/train_image/jpg'
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = LoadImage_LeNet5(datapath, transform)
    return dataset

def load_pid_test_data():
    '''rootpath = '/Data/zkx/piddata/test_data/image/test_image/jpg'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    shutil.rmtree(rootpath)
    os.makedirs(rootpath)

    save_stict = '/Data/zkx/piddata/test_data/image/test_image/jpg/stiction'
    save_nonstict = '/Data/zkx/piddata/test_data/image/test_image/jpg/nonstiction'
    os.makedirs(save_stict)
    os.makedirs(save_nonstict)

    # Remove images to target dirs
    stictionpath = '/Data/zkx/piddata/test_data/image/differspan/span2'
    filelist = os.listdir(stictionpath)
    for i in range(len(filelist)):
        impath = os.path.join(stictionpath, filelist[i])
        im = Image.open(impath)
        newfilename = filelist[i][0:-4] + '.jpg'
        savepath = os.path.join(save_stict, newfilename)
        rgb_im = im
        rgb_im.save(savepath)

    nonstictionpath = '/Data/zkx/piddata/test_data/image/samespan/nonstiction_200'
    filelist = os.listdir(nonstictionpath)
    for i in range(len(filelist)):
        impath = os.path.join(nonstictionpath, filelist[i])
        im = Image.open(impath)
        newfilename = filelist[i][0:-4] + '.jpg'
        savepath = os.path.join(save_nonstict, newfilename)
        rgb_im = im
        rgb_im.save(savepath)'''

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testdatapath = '/home/zhangkexin/code/work1/finalresult/lenet5/test_image/jpg'
    dataset = LoadImage_LeNet5(testdatapath, transform)

    return dataset

def lenet5_train(epochs=8, train_batch=16):
    # Para
    train_batch_size = train_batch
    epochs = epochs
    loss_display = 10
    # Load Dataset
    train_data = load_pid_train_data()


    # Load DataLoader
    trainloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, drop_last=True)

    # Model
    net = lenet5()
    # optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    # Step1: Training CNN model
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % loss_display == loss_display - 1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / loss_display))
                running_loss = 0.0
    print('Finished Training LeNet5')
    # Save model
    PATH = '/home/zhangkexin/code/work1/result/lenet5/model_lenet5.pth'
    torch.save(net.state_dict(), PATH)

def lenet5_test(testbatch=60):
    PATH = '/home/zhangkexin/code/work1/finalresult/lenet5/model_lenet5.pth'
    finalnet = lenet5()
    finalnet.load_state_dict(torch.load(PATH))
    test_data = load_pid_test_data()
    testloader = DataLoader(test_data, batch_size=testbatch, shuffle=False, drop_last=False)
    correct = 0
    total = 0
    label = []
    pred = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = finalnet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            la = labels.cpu().numpy()
            pr  = predicted.cpu().numpy()
            label.append(la)
            pred.append(pr)

    Test_Y = np.array(label)
    pred_label = np.array(pred)

    return Test_Y[0], pred_label[0]

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

    #lenet5_train(epochs=8, train_batch=16)
    Test_Y, pred_label = lenet5_test(testbatch=60)
    get_metrics(Test_Y, pred_label)
    #print(Test_Y)
    #print(pred_label)



