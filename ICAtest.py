# -*- coding: utf-8 -*-
"""halp.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jLw4_7EQsbIKzIiBzDP7drtdo4l1L_Fv
"""

pip install umap-learn

# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1o8dq_jQXTDcSGwBfFnCEYI-EIxF_jnG0
"""

# CNN LSTM Linear Classifier
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import random
import umap
from scipy import signal
from sklearn.decomposition import PCA, FastICA
from torch.utils.data import DataLoader, sampler, TensorDataset 

def power(data,label):
  N=label.shape[0]
  channel=data.shape[1]
  pointer=0
  newlabel=np.zeros(1)
  powerdata=np.zeros(channel)
  for i in range(N):
    if (i+1)==N:
      newlabel=np.hstack((newlabel,label[i]))
      temp=data[pointer:i,:]
      temp=np.linalg.norm(temp,axis=0)
      powerdata=np.vstack((powerdata,temp**2/(i-pointer)))
    elif label[i+1]!=label[i]:
      newlabel=np.hstack((newlabel,label[i]))
      temp=data[pointer:i,:]
      temp=np.linalg.norm(temp,axis=0)
      powerdata=np.vstack((powerdata,temp**2/(i-pointer)))
      pointer=i
  data=powerdata[1:,:]
  label=newlabel[1:]
  data,label=torch.from_numpy(data).float(),torch.from_numpy(label).long()
  return data,label

def uniform(data):
  um=umap.UMAP(n_components=30,n_neighbors=15,min_dist=0.1)
  data=um.fit_transform(data)
  return data

def principle(data):
  pca=PCA(n_components=30)
  data=pca.fit_transform(data)
  return data


def data_process(data,group):
  data=data['data']
  group=group['stim']
  #b,a = signal.butter(4, (80.1,80.5),'bandpass',fs=1000)
  #data = signal.filtfilt(b,a,data.T)
  data = data
  index,_=np.where(group>0)
  train = data[index,:]
  trainlabel=group[index,0]
  # Channel delete
  #train=uniform(train)
  trainlabel=trainlabel-1
  train,trainlabel=torch.from_numpy(train).float(),torch.from_numpy(trainlabel).long()
  return train[30000:],trainlabel[30000:]

class myNetwork(nn.Module):
    def __init__(self, channel ,hidden_size, num_states):
        super().__init__()
        # CNN
        '''
        self.conv1= nn.Conv1d(channel,32,20,stride=1)
        self.conv2= nn.Conv1d(32,64,10)
        self.conv3= nn.Conv1d(64,128,8)
        self.flat= nn.Flatten()
        
        # RNN
        
        self.rnn=nn.LSTM(input_size=channel,
                         hidden_size=hidden_size,
                         num_layers=1,
                         batch_first=True,
                         )
        '''
        # linear layer
        self.bn1 = nn.BatchNorm1d(channel)
        self.fc1 = nn.Linear(channel,hidden_size)
        #self.dp1 = nn.Dropout(0.25)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        #self.dp2 = nn.Dropout(0.25)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size,hidden_size)
        #self.dp3 = nn.Dropout(0.25)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size,hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.fc5 = nn.Linear(hidden_size,num_states)

    def forward(self, x):
        # CNN
        '''
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.flat(x)
        
        # RNN
        
        rout,(h_n,h_c)=self.rnn(x,None)
        x = self.bn2(rout[:,-1,:])
        '''
        x = self.bn1(x)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dp1(x)
        x = self.bn2(x)
        x = self.fc2(x)
        x = F.relu(x)
        #x = self.dp2(x)
        x = self.bn3(x)
        x = self.fc3(x)
        x = F.relu(x)
        #x = self.dp3(x)
        x = self.bn4(x)
        x = self.fc4(x)
        x = self.bn5(x)
        scores = self.fc5(x)
        return scores

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight,nonlinearity='relu')

# Load data and Preprocess 
data = scipy.io.loadmat('drive/MyDrive/train/bp_data.mat')
group = scipy.io.loadmat('drive/MyDrive/train/bp_stim.mat')
bp,bplabel=data_process(data,group)
data = scipy.io.loadmat('drive/MyDrive/train/ht_data.mat')
group = scipy.io.loadmat('drive/MyDrive/train/ht_stim.mat')
ht,htlabel=data_process(data,group)
data = scipy.io.loadmat('drive/MyDrive/train/jp_data.mat')
group = scipy.io.loadmat('drive/MyDrive/train/jp_stim.mat')
jp,jplabel=data_process(data,group)
data = scipy.io.loadmat('drive/MyDrive/train/wc_data.mat')
group = scipy.io.loadmat('drive/MyDrive/train/wc_stim.mat')
wc,wclabel=data_process(data,group)
data = scipy.io.loadmat('drive/MyDrive/train/zt_data.mat')
group = scipy.io.loadmat('drive/MyDrive/train/zt_stim.mat')
zt,ztlabel=data_process(data,group)
data = scipy.io.loadmat('drive/MyDrive/train/jc_data.mat')
group = scipy.io.loadmat('drive/MyDrive/train/jc_stim.mat')
jc,jclabel=data_process(data,group)
data = scipy.io.loadmat('drive/MyDrive/train/mv_data.mat')
group = scipy.io.loadmat('drive/MyDrive/train/mv_stim.mat')
test,testlabel=data_process(data,group)

def ICA(data):
  ica = FastICA(n_components=20,tol=.0001, max_iter = 200)
  data = ica.fit_transform(data)
  return data

plt.plot(range(ht.shape[0]),ht[:,0])

plt.plot(range(bp.shape[0]-30000),bplabel[30000:]);

transform = ICA(ht)

transform2 = ICA(ht)

from scipy.fft import fft, fftfreq
N = ht.shape[0]
# sample spacing
T = 1.0/1000.0
y=ht.cpu().detach().numpy()[:,0]

#from scipy.signal import blackman
#w = blackman(N)
print(jp.shape)
yf = fft(jp.cpu().detach().numpy()[:,0])
#yf=np.zeros([yf.shape[0],20])
#for i in range(20):
 # yf[:,i] = fft(transform[:,i])
xf = fftfreq(N, T)[:N//2]
import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()

print(xf.shape)
print(yf.shape)

print(xf[np.argmax(yf[0:N//2])])

print(transform.shape)

plt.plot(range(ht.shape[0]),transform[:,3]);

plt.plot(range(ht.shape[0]),transform2[:,]);

print(bp[100300].shape)

train=torch.vstack((bp,ht,jp,wc,zt,jc))
trainlabel=torch.hstack((bplabel,htlabel,jplabel,wclabel,ztlabel,jclabel))
# Create Dataset and dataloader
test_ds= TensorDataset(test, testlabel)
train_ds = TensorDataset(train, trainlabel)

#If a batch in BatchNorm only has 1 sample it wont work, so dropping the last in case that happens
train_dl = DataLoader(train_ds, batch_size=400000, shuffle=True, drop_last=True)
test_dl =  DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, drop_last=True)

channel=30
'''
# powerval
train,trainlabel=power(train,trainlabel)
test,testlabel=power(test,testlabel)

# RNN random
path = 3
for recurrent in range(path):
  pick=np.random.choice(channel,channel,replace='False')
  train=torch.hstack((train,train[:,pick]))
  test=torch.hstack((test,test[:,pick]))
train=train.view(train.shape[0],path+1,-1)
test=test.view(test.shape[0],path+1,-1)


# CNN reshape
train=train.view(train.shape[0],channel,-1)
test=test.view(test.shape[0],channel,-1)

# past time
column=train
col=test
for i in range(49):
  column=torch.roll(column,1, 0)
  column[0,:,0]=0
  col=torch.roll(col,1, 0)
  col[0,:,0]=0
  train=torch.dstack((train,column))
  test=torch.dstack((test,col))
'''
# parameter & loss function & optimize
learning_rate = 0.01
decay=0.001
hidden=400
outlayer=5

theNetwork=myNetwork(channel,hidden,outlayer)
theNetwork.apply(init_normal)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(theNetwork.parameters(),lr=learning_rate,weight_decay=decay)
stop=1
# start training
while stop!=0:
  for x,y in train_dl: # batch of training points
    optimizer.zero_grad()
    predict=theNetwork(x)
    loss=criterion(predict,y)
    loss.backward()
    optimizer.step()
  # valid per epoch
  output=theNetwork(test)
  _,predict=torch.max(output.data,1)
  correct=(predict==testlabel).sum()/len(testlabel)
  print(stop,loss.item(),correct)
  stop=stop+1