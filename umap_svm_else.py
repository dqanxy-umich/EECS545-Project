#Programmer: Yichen Ding
#Date: Apr 2022

import scipy.io
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import sklearn
from sklearn.decomposition import PCA
from scipy import signal

def preprocessing(X, y, C_=0, pooling=False, normalize=False):
  '''
  Slice X into matrices, each is a continuous wave with same label. 
  All data with label -2,-1,0 are discarded. 
  C_ is the new dimension for PCA/UMAP
  Returns:
  X_new, a list of (*, C) 2d-arrays, where * varies if not padded;
  y_new, the corresponding labels;
  change, indices where the reduced y changes value. 
  '''
  N,C = X.shape
  shape_finder = lambda arr: arr.shape[0]
  
  #Extract the top boundary shape
  from scipy.signal import argrelextrema
  x_e = [X.T[i][argrelextrema(X.T[i], np.greater, axis=0)[0]] 
         for i in range(C)]
  min_ind = min(list(map(shape_finder, x_e)))
  X_e = np.array(list(map(lambda arr: arr[:min_ind], x_e))).T
  
  y_cutoff_ind = (np.arange(min_ind)*(N/min_ind)+0.5*N/min_ind).astype(int)
  y_e = y[y_cutoff_ind]
  
  index,_=np.where(y_e>0)
  X_e = X_e[index,:]
  y_e = y_e[index,:]
    
  if pooling:
    X_e = X_e[::2,:]
    y_e = y_e[::2,:]
  
  #PCA/UMAP here, not needed for modelling with a single patient. 
  '''
  if C_ > 0: 
    pca=PCA(n_components = C_)
    X_e=pca.fit_transform(X_e)
  '''
  if C_>0:
    X_e = uniform(X_e, C_)
  
  #Normalize
  if normalize:
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(X_e.T)
    X_e = scaler.transform(X_e.T).T
  
  return X_e, y_e
  
def preprocessing_p2(X_e, y_e, padding=True):
  #Cutting the dataset into epochs
  shape_finder = lambda arr: arr.shape[0]
  M = X_e.shape[0]
  yy = np.array(y_e).reshape((M,))
  change = np.where(yy[:-1]!=yy[1:])[0]+1
  y_new = [yy[change[i]] 
           for i in range(change.shape[0]-1)
          ]
  X_new = [X_e[change[i]:change[i+1],:] 
           for i in range(change.shape[0]-1)
          ]
  
  #Padding with 0, to get same length for each chunk
  if padding:
    max_size = max(list(map(shape_finder, X_new)))
    X_new = np.array([np.pad(i,
                             ((0,max_size-shape_finder(i)),(0,0)),
                             'constant',
                             constant_values=0) 
                      for i in X_new
                     ])
    
  return X_new, y_new

import umap.umap_ as umap
def uniform(data, C_):
  um = umap.UMAP(n_components=C_,n_neighbors=15,min_dist=0.1)
  data=um.fit_transform(data)
  return data

def conv_out_size(slen, kernel_size, stride):
    return int((slen - kernel_size) / stride + 1)

################Modified from another course's provided code################
from torch import nn
class MultiClassConvNet(torch.nn.Module):
    def __init__(self, side_length, conv_channels_1, conv_channels_2, linear_hidden, num_classes):
        super().__init__()
        linear_in_side = int(conv_out_size(conv_out_size(side_length, 3, 1)/2, 3, 1)/2)
        self.net = nn.Sequential(
            nn.Conv1d(10, conv_channels_1, kernel_size=3), 
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(conv_channels_1, conv_channels_2, kernel_size=3),
            nn.MaxPool1d(kernel_size=2),
            #nn.Flatten(start_dim=1), 
            nn.Linear(conv_channels_2 * linear_in_side, linear_hidden), 
            nn.ReLU(), 
            nn.Linear(linear_hidden, out_features=num_classes),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, x):
        return self.net(x)
convnet = MultiClassConvNet(side_length=1326, conv_channels_1=16, conv_channels_2=24, linear_hidden=64, num_classes=5)

from tqdm import tqdm
convnet_optimizer = torch.optim.Adam(convnet.parameters())
num_epochs = 128

def train_loop(model, transform_fn, loss_fn, optimizer, dataloader, num_epochs):
    tbar = tqdm(range(num_epochs))
    for _ in tbar:
        loss_total = 0.
        for i, (x, y) in enumerate(dataloader):
            x = transform_fn(x)
            pred = model(x)
            loss = loss_fn(pred, y.squeeze(-1))
            ## Parameter updates
            model.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
        tbar.set_description(f"Train loss: {loss_total/len(dataloader)}")
        
    return loss_total/len(dataloader)

def calculate_test_accuracy(model, transform_fn, test_dataloader):
    y_true = []
    y_pred = []
    tf = nn.Flatten()
    for (xi, yi) in test_dataloader:
        xi = transform_fn(xi)
        pred = model(xi)
        yi_pred = pred.argmax(-1)
        y_true.append(yi)
        y_pred.append(yi_pred)
    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    accuracy = (y_true == y_pred).float().mean()
    return accuracy
############################################################################

data = scipy.io.loadmat('train/bp_data.mat')['data']
group = scipy.io.loadmat('train/bp_stim.mat')['stim']
bp,bplabel=preprocessing(data,group,C_=10,pooling=False)
data = scipy.io.loadmat('train/ht_data.mat')['data']
group = scipy.io.loadmat('train/ht_stim.mat')['stim']
ht,htlabel=preprocessing(data,group,C_=10,pooling=False)
data = scipy.io.loadmat('train/jp_data.mat')['data']
group = scipy.io.loadmat('train/jp_stim.mat')['stim']
jp,jplabel=preprocessing(data,group,C_=10,pooling=False)
data = scipy.io.loadmat('train/wc_data.mat')['data']
group = scipy.io.loadmat('train/wc_stim.mat')['stim']
wc,wclabel=preprocessing(data,group,C_=10,pooling=False)
data = scipy.io.loadmat('train/zt_data.mat')['data']
group = scipy.io.loadmat('train/zt_stim.mat')['stim']
zt,ztlabel=preprocessing(data,group,C_=10,pooling=False)
data = scipy.io.loadmat('train/jc_data.mat')['data']
group = scipy.io.loadmat('train/jc_stim.mat')['stim']
jc,jclabel=preprocessing(data,group,C_=10,pooling=False)
data = scipy.io.loadmat('train/mv_data.mat')['data']
group = scipy.io.loadmat('train/mv_stim.mat')['stim']
mv,mvlabel=preprocessing(data,group,C_=10,pooling=False)
data = scipy.io.loadmat('train/wm_data.mat')['data']
group = scipy.io.loadmat('train/wm_stim.mat')['stim']
wm,wmlabel=preprocessing(data,group,C_=10,pooling=False)

X_train_raw = np.concatenate((bp,ht,jp,wc,zt,jc), axis=0)
y_train_raw = np.concatenate((bplabel,htlabel,jplabel,wclabel,ztlabel,jclabel), axis=0)
X_train_cut, y_train_cut = preprocessing_p2(X_train_raw, y_train_raw, padding=True)

X_test_extra = np.concatenate((mv,wm), axis=0)
y_test_extra = np.concatenate((mvlabel,wmlabel), axis=0)

from sklearn.model_selection import train_test_split
#For kNN, and SVM
X_train, X_test, y_train, y_test = train_test_split(X_train_raw, y_train_raw, test_size=0.3, random_state=42)

#For CNN
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_train_cut, y_train_cut, test_size=0.3, random_state=42)

X_t_train = torch.from_numpy(X_train_c)
y_t_train = torch.tensor(y_train_c)
X_t_test = torch.from_numpy(X_test_c)
y_t_test = torch.tensor(y_test_c)

from torch.utils.data import Dataset, DataLoader, TensorDataset 
from torchvision import datasets
from torchvision.transforms import ToTensor
train_ds = TensorDataset(X_t_train, y_t_train.type(torch.LongTensor))
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
test_ds = TensorDataset(X_t_test, y_t_test.type(torch.LongTensor))
test_dl = DataLoader(test_ds, batch_size=80, shuffle=True, drop_last=True)

train_loop(convnet, lambda x: x, nn.NLLLoss(), convnet_optimizer, train_dl, num_epochs)
print('Train CNN: ', calculate_test_accuracy(convnet, lambda x: x, train_dl))
print('Test CNN: ', calculate_test_accuracy(convnet, lambda x: x, test_dl))

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train.ravel())
print('Train kNN: ', neigh.score(X_train, y_train))
print('Test kNN: ', neigh.score(X_test, y_test))
print('Test kNN extra: ', neigh.score(X_test_extra, y_test_extra))

from sklearn.svm import SVC
svc_clf = SVC(gamma='auto')
svc_clf.fit(X_train, y_train.ravel())
print('Train SVM: ', svc_clf.score(X_train, y_train.ravel()))
print('Test SVM: ', svc_clf.score(X_test, y_test.ravel()))
print('Test SVM extra: ', svc_clf.score(X_test_extra, y_test_extra.ravel()))

X_test_extra = np.concatenate((mv,wm), axis=0)
y_test_extra = np.concatenate((mvlabel,wmlabel), axis=0)