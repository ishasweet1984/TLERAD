import sys,os
import tqdm
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import glob
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler




# In[2]:

feature_file_name = sys.argv[1]
label_file_name = sys.argv[2]
classify_category = int(sys.argv[3])
if os.path.isfile('features/'+feature_file_name):
    features_df = pd.read_csv('features/'+feature_file_name,header=None).set_index(0)
else:
    print("No such feature files in directory features")
if os.path.isfile('label/'+label_file_name):
    label_dict = pd.read_csv('label/'+label_file_name,header=None).set_index(0).to_dict()[1]
else:
    print("No such feature files in directory label")
X = features_df.values
y = np.array([label_dict[i] for i in features_df.index])
num_class = len(set(y))
Kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=1337)
sp_idx = [pair for pair in Kfold.split(X,y)]
X.shape,y.shape




# In[3]:


RF_proba = np.zeros((len(X),classify_category))
lgb_param = {
    'boosting_type': 'gbdt',
    #'objective': 'multiclass',
    'learning_rate': 0.1,
    'num_leaves': 300,
    'max_depth': 2,
    'subsample': 0.5,
    'colsample_bytree': 0.5,
    'n_estimators': 600,
    'n_jobs': 30,
    'random_state':1337,
}
LGB_proba = np.zeros((len(X),classify_category))
xgb_param = {
    'max_depth':5,
    'n_estimators':200,
    'learning_rate':0.1,
    'subsample':0.5,
    'colsample_bytree':0.5,
    'min_child_weight':3,
    'objective':'multi:softmax',
    'eval_metric':'mlogloss',
    #'tree_method':'gpu_hist',
    'n_jobs':32,
    'seed':1337,
    'num_class':num_class}
XGB_proba = np.zeros((len(X),classify_category))
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
scaler = StandardScaler()
scaler.fit(X)
pt_X = scaler.transform(X)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(pt_X.shape[1], 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_class)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
EPOCH = 20
BATCH_SIZE = 64
PyT_proba = np.zeros((len(X),classify_category))


# In[4]:


model = RandomForestClassifier(max_depth = 30,n_estimators = 200,n_jobs = 32)
model.fit(X,y)
with open('model/RF_model.pickle','wb') as f:
    pickle.dump(model,f)
model = lgb.LGBMClassifier(**lgb_param)
model.fit(X,y)
with open('model/LGB_model.pickle','wb') as f:
    pickle.dump(model,f)
model = XGBClassifier(**xgb_param)
model.fit(X,y)
with open('model/XGB_model.pickle','wb') as f:
    pickle.dump(model,f)
ts_train_x = torch.tensor(pt_X).float()
ts_train_y = torch.tensor(y).type(torch.LongTensor)
torch_dataset = Data.TensorDataset(ts_train_x, ts_train_y)
train_loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=1,              # subprocesses for loading data
)

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.00001)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
torch.save(model.state_dict(), 'model/PTH_model.pickle')
with open('model/PTH_scaler.pickle','wb') as f:
    pickle.dump(scaler,f)


# # Stack Model

# In[5]:


st_X = np.concatenate((RF_proba,LGB_proba,XGB_proba,PyT_proba),axis=1)
stxgb_param = {
    'gpu_id' : 3,
    'max_depth' : 2,
    'n_estimators' : 80,
    'learning_rate' : 0.1,
    'objective' : 'multi:softmax',
    'eval_metric' : 'mlogloss',
    #'tree_method' : 'gpu_hist',
    'n_jobs' : 32,
    'seed' : 1337,
    'num_class' :num_class}
model = XGBClassifier(**stxgb_param)
model.fit(st_X,y)
with open('model/ST_model.pickle','wb') as f:
    pickle.dump(model,f)


# In[ ]:




